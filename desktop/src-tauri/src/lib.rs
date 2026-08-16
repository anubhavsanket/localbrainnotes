//! LocalBrain desktop shell.
//!
//! Responsibilities:
//!   1. On startup, spawn the Python FastAPI backend as a local sidecar
//!      (`backend/main.py`) on 127.0.0.1:8000 and wait for `/api/health`.
//!   2. Open the webview window hosting the React chat UI.
//!   3. On exit, kill the backend child process.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent};

/// Holds the spawned backend child so we can kill it on exit.
struct BackendChild(Mutex<Option<Child>>);

/// Repo root = two levels above `src-tauri/`:
/// CARGO_MANIFEST_DIR = `<repo>/desktop/src-tauri` → `<repo>`.
fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("CARGO_MANIFEST_DIR has no grandparent")
        .to_path_buf()
}

/// Locate the venv python on Windows: `<root>/.venv/Scripts/python.exe`.
fn python_bin() -> PathBuf {
    project_root().join(".venv").join("Scripts").join("python.exe")
}

fn backend_main() -> PathBuf {
    project_root().join("backend").join("main.py")
}

fn backend_alive() -> bool {
    // Cheap TCP probe — mirrors the health route without pulling in HTTP deps.
    use std::net::TcpStream;
    matches!(
        TcpStream::connect_timeout(
            &"127.0.0.1:8000".parse().expect("static addr"),
            Duration::from_millis(300),
        ),
        Ok(_)
    )
}

/// Spawn `python backend/main.py`, then poll until `/api/health` responds.
fn spawn_backend() -> Option<Child> {
    let py = python_bin();
    let main = backend_main();
    if !py.exists() {
        eprintln!("[localbrain] venv python not found at {:?}", py);
        return None;
    }
    if !main.exists() {
        eprintln!("[localbrain] backend/main.py not found at {:?}", main);
        return None;
    }

    let mut child = Command::new(&py)
        .arg(&main)
        .current_dir(project_root())
        // Disable uvicorn auto-reload: the reloader spawns a subprocess that
        // would outlive the shell and keep port 8000 bound after exit.
        .env("LOCALBRAIN_DISABLE_RELOAD", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    // Wait up to ~30s for the server to come up.
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if backend_alive() {
            return Some(child);
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    eprintln!("[localbrain] backend did not become healthy in 30s");
    let _ = child.kill();
    None
}

fn kill_child(app: &tauri::AppHandle) {
    if let Some(state) = app.try_state::<BackendChild>() {
        if let Some(mut child) = state.0.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub fn run() {
    let app = tauri::Builder::default()
        .setup(|app| {
            app.manage(BackendChild(Mutex::new(spawn_backend())));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building LocalBrain tauri application");

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            kill_child(app_handle);
        }
        if let RunEvent::Exit = event {
            kill_child(app_handle);
        }
    });
}
