use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

pub struct ServerProcess {
    child: Child,
    pub socket_path: String,
}

impl ServerProcess {
    /// Spawn darktable-server and read the SOCKET= line from stdout.
    pub fn start(server_bin: &str, configdir: &str) -> Result<Self, String> {
        let mut child = Command::new(server_bin)
            .args(["--core", "--configdir", configdir])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("failed to spawn darktable-server: {e}"))?;

        let stdout = child
            .stdout
            .take()
            .ok_or("failed to capture server stdout")?;

        let reader = BufReader::new(stdout);
        let mut socket_path = None;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("failed to read server stdout: {e}"))?;
            if let Some(path) = line.strip_prefix("SOCKET=") {
                socket_path = Some(path.to_string());
                break;
            }
        }

        let socket_path =
            socket_path.ok_or("server exited without printing SOCKET= line")?;

        eprintln!("[tauri] server started, socket: {socket_path}");

        Ok(Self {
            child,
            socket_path,
        })
    }

    /// Send SIGTERM to the server process.
    pub fn stop(&mut self) {
        #[cfg(unix)]
        {
            unsafe {
                libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM);
            }
            // Give it a moment to shut down
            let _ = self.child.wait();
        }
        #[cfg(not(unix))]
        {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        self.stop();
    }
}
