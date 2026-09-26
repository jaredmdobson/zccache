//! zccache#1652: the embedded service must persist `metadata.bin` so the
//! next start restores the stat-verified hash fast path instead of starting
//! empty every time.
//!
//! Each test drives a real compile through the public facade, ends the
//! service (graceful shutdown, or a drop without shutdown), and then asserts
//! the snapshot exists on disk, decodes, and contains the compiled source.

use super::*;
use std::path::PathBuf;
use tempfile::TempDir;

async fn start_service(cache_root: &std::path::Path) -> ZccacheService {
    let mut audit = AuditConfig::default();
    audit.mode = crate::audit::AuditMode::Off;
    ZccacheService::start(ZccacheConfig {
        host: HostIdentity {
            product: "metadata-persist-test".into(),
            instance_id: uuid::Uuid::new_v4().to_string(),
            workspace_id: "metadata-persist-workspace".into(),
        },
        cache_root: cache_root.into(),
        audit,
        limits: ServiceLimits::default(),
        runtime: RuntimeHooks::default(),
        cancellation: None,
    })
    .await
    .expect("service start")
}

struct Fixture {
    _temp: TempDir,
    cache_root: PathBuf,
    source: PathBuf,
    request: CompileRequest,
}

fn fixture(compiler: crate::core::NormalizedPath) -> Fixture {
    let temp = TempDir::new().expect("tempdir");
    let cache_root = temp.path().join("cache");
    let work = temp.path().join("work");
    std::fs::create_dir_all(&work).expect("work dir");
    let header = work.join("persist.h");
    let source = work.join("persist.cpp");
    let object = work.join("persist.o");
    std::fs::write(&header, "#define PERSIST_VALUE 11\n").expect("header");
    std::fs::write(
        &source,
        "#include \"persist.h\"\nint persist(void) { return PERSIST_VALUE; }\n",
    )
    .expect("source");
    let request = CompileRequest {
        audit: AuditContext::new(
            crate::audit::AuditId::new("metadata-persist-run").expect("id"),
            crate::audit::AuditId::new("metadata-persist-trace").expect("id"),
        ),
        compiler,
        args: vec![
            "-c".into(),
            source.to_string_lossy().into_owned(),
            "-o".into(),
            object.to_string_lossy().into_owned(),
        ],
        cwd: work.into(),
        env: Vec::new(),
        stdin: Vec::new(),
    };
    Fixture {
        _temp: temp,
        cache_root,
        source,
        request,
    }
}

/// Assert `metadata.bin` exists under the cache root, decodes at the current
/// format, and carries a hash for the compiled source.
fn assert_metadata_persisted(cache_root: &std::path::Path, source: &std::path::Path) {
    let path = crate::core::config::metadata_path_from_cache_dir(
        &crate::core::NormalizedPath::new(cache_root),
    );
    assert!(
        path.exists(),
        "metadata.bin must be persisted under the embedded cache root: {}",
        path.display()
    );
    let restored =
        crate::fscache::MetadataCache::load_from_disk(path.as_path()).expect("metadata.bin loads");
    assert!(
        restored
            .get_cached_hash(&crate::core::NormalizedPath::new(source))
            .is_some(),
        "restored metadata must carry the compiled source's hash ({} entries)",
        restored.len()
    );
}

async fn compile_once(service: &ZccacheService, request: CompileRequest) {
    let response = service.compile(request).await.expect("compile");
    assert_eq!(
        response.exit_code,
        0,
        "compiler stderr: {}",
        String::from_utf8_lossy(&response.stderr)
    );
}

/// A host can stop after the artifact checkpoint but before the depgraph and
/// metadata snapshots. The restored old graph/hash must validate inputs before
/// it can select an artifact from the newer index.
#[tokio::test]
async fn partial_flush_with_new_index_and_old_graph_serves_current_inputs() {
    let Some(compiler) = crate::test_support::find_clang() else {
        return;
    };
    let fx = fixture(compiler);
    let first = start_service(&fx.cache_root).await;
    let original = first
        .compile(fx.request.clone())
        .await
        .expect("first compile");
    assert_eq!(original.exit_code, 0);
    let object = fx.source.with_extension("o");
    let original_bytes = std::fs::read(&object).expect("first object");
    let state_root = first.stats().await.expect("first stats").cache_root;
    first
        .shutdown(ShutdownMode::Graceful)
        .await
        .expect("first shutdown");

    let cache_dir = crate::core::NormalizedPath::new(&state_root);
    let depgraph_path =
        crate::core::config::depgraph_dir_from_cache_dir(&cache_dir).join("depgraph.bin");
    let metadata_path = crate::core::config::metadata_path_from_cache_dir(&cache_dir);
    let compiler_hash_path =
        crate::core::config::compiler_hash_cache_path_from_cache_dir(&cache_dir);
    let system_includes_path =
        crate::core::config::system_includes_cache_path_from_cache_dir(&cache_dir);
    let old_depgraph = std::fs::read(&depgraph_path).expect("old depgraph");
    let old_metadata = std::fs::read(&metadata_path).expect("old metadata");
    let old_compiler_hash = std::fs::read(&compiler_hash_path).ok();
    let old_system_includes = std::fs::read(&system_includes_path).ok();

    std::fs::write(
        fx.source.parent().unwrap().join("persist.h"),
        "#define PERSIST_VALUE 222\n",
    )
    .expect("change input");
    let second = start_service(&fx.cache_root).await;
    let changed = second
        .compile(fx.request.clone())
        .await
        .expect("changed compile");
    assert_eq!(changed.exit_code, 0);
    assert!(
        !changed.cached,
        "changed header must invalidate the old hit"
    );
    let changed_bytes = std::fs::read(&object).expect("changed object");
    assert_ne!(original_bytes, changed_bytes);
    second
        .shutdown(ShutdownMode::Graceful)
        .await
        .expect("second shutdown");

    // Model the interruption boundary: index.bin and artifact payloads are
    // from the second compile, while the other snapshots are from the first.
    // A partial flush leaves exactly these generations on disk.
    std::fs::write(&depgraph_path, old_depgraph).expect("restore old depgraph");
    std::fs::write(&metadata_path, old_metadata).expect("restore old metadata");
    for (path, prior) in [
        (&compiler_hash_path, old_compiler_hash),
        (&system_includes_path, old_system_includes),
    ] {
        match prior {
            Some(bytes) => std::fs::write(path, bytes).expect("restore old snapshot"),
            None if path.exists() => std::fs::remove_file(path).expect("restore absent snapshot"),
            None => {}
        }
    }
    std::fs::remove_file(&object).expect("remove output before restart");

    let restarted = start_service(&fx.cache_root).await;
    let response = restarted
        .compile(fx.request.clone())
        .await
        .expect("compile after partial flush");
    assert_eq!(response.exit_code, 0);
    assert_eq!(
        std::fs::read(&object).expect("restored object"),
        changed_bytes
    );
    restarted
        .shutdown(ShutdownMode::Graceful)
        .await
        .expect("restart shutdown");
}

#[tokio::test]
async fn graceful_shutdown_persists_metadata_snapshot() {
    let Some(compiler) = crate::test_support::find_clang() else {
        return;
    };
    let fx = fixture(compiler);
    let service = start_service(&fx.cache_root).await;
    compile_once(&service, fx.request.clone()).await;
    let state_root = service.stats().await.expect("stats").cache_root;
    let report = service
        .shutdown_detailed(ShutdownMode::Graceful)
        .await
        .expect("shutdown");
    assert!(report.flushed.is_complete(), "{report:?}");
    assert_metadata_persisted(state_root.as_path(), &fx.source);

    // The next start must restore it instead of starting empty.
    let restarted = start_service(&fx.cache_root).await;
    let stats = restarted.stats().await.expect("stats");
    assert!(
        stats.metadata_entries > 0,
        "restarted service must restore the persisted metadata snapshot"
    );
    restarted
        .shutdown(ShutdownMode::Graceful)
        .await
        .expect("restart shutdown");
}

#[tokio::test]
async fn drop_without_shutdown_persists_metadata_snapshot() {
    let Some(compiler) = crate::test_support::find_clang() else {
        return;
    };
    let fx = fixture(compiler);
    let state_root;
    {
        let service = start_service(&fx.cache_root).await;
        compile_once(&service, fx.request.clone()).await;
        state_root = service.stats().await.expect("stats").cache_root;
        drop(service);
    }
    assert_metadata_persisted(state_root.as_path(), &fx.source);
}
