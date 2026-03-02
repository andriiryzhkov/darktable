use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

pub struct IpcClient {
    stream: UnixStream,
    next_id: AtomicU64,
}

impl IpcClient {
    /// Connect to the darktable-server Unix socket.
    pub async fn connect(path: &str) -> Result<Self, String> {
        let stream = UnixStream::connect(path)
            .await
            .map_err(|e| format!("failed to connect to socket {path}: {e}"))?;

        Ok(Self {
            stream,
            next_id: AtomicU64::new(1),
        })
    }

    /// Send a JSON-RPC request and return the result.
    pub async fn request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let id_str = format!("req-{id}");

        let request = serde_json::json!({
            "id": id_str,
            "method": method,
            "params": params,
        });

        self.write_frame(&request).await?;

        // Read responses, skipping events (id == null)
        loop {
            let response = self.read_frame().await?;

            // Check if this is an event (id is null)
            if response.get("id").and_then(|v| v.as_str()).is_none()
                && response.get("event").is_some()
            {
                // Skip events for now — just log them
                eprintln!(
                    "[ipc] event: {}",
                    response
                        .get("event")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                );
                continue;
            }

            // Check for error
            if let Some(err) = response.get("error") {
                if !err.is_null() {
                    let msg = err
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown error");
                    return Err(msg.to_string());
                }
            }

            // Return the result
            return Ok(response
                .get("result")
                .cloned()
                .unwrap_or(serde_json::Value::Null));
        }
    }

    /// Write a length-prefixed JSON frame to the socket.
    async fn write_frame(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let json = serde_json::to_string(value)
            .map_err(|e| format!("JSON serialization failed: {e}"))?;
        let bytes = json.as_bytes();
        let len = bytes.len() as u32;

        self.stream
            .write_all(&len.to_be_bytes())
            .await
            .map_err(|e| format!("failed to write frame length: {e}"))?;

        self.stream
            .write_all(bytes)
            .await
            .map_err(|e| format!("failed to write frame data: {e}"))?;

        Ok(())
    }

    /// Read a length-prefixed JSON frame from the socket.
    async fn read_frame(&mut self) -> Result<serde_json::Value, String> {
        let mut len_buf = [0u8; 4];
        self.stream
            .read_exact(&mut len_buf)
            .await
            .map_err(|e| format!("failed to read frame length: {e}"))?;

        let len = u32::from_be_bytes(len_buf) as usize;
        if len == 0 || len > 16 * 1024 * 1024 {
            return Err(format!("invalid frame length: {len}"));
        }

        let mut buf = vec![0u8; len];
        self.stream
            .read_exact(&mut buf)
            .await
            .map_err(|e| format!("failed to read frame data: {e}"))?;

        serde_json::from_slice(&buf)
            .map_err(|e| format!("JSON parse failed: {e}"))
    }
}
