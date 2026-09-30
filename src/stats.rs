// SPDX-License-Identifier: Apache-2.0
//! Process-wide counters for the /stack endpoint.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

pub struct Stats {
    counters: Mutex<HashMap<String, u64>>,
    started: Instant,
}

impl Stats {
    pub fn new() -> Self {
        Self { counters: Mutex::new(HashMap::new()), started: Instant::now() }
    }
    pub fn incr(&self, key: &str) {
        self.add(key, 1);
    }
    pub fn add(&self, key: &str, n: u64) {
        *self.counters.lock().unwrap().entry(key.to_string()).or_insert(0) += n;
    }
    pub fn snapshot(&self) -> serde_json::Value {
        let c = self.counters.lock().unwrap();
        let mut grouped: HashMap<String, serde_json::Map<String, serde_json::Value>> = HashMap::new();
        for (k, v) in c.iter() {
            let (comp, metric) = k.split_once('.').unwrap_or(("misc", k.as_str()));
            grouped.entry(comp.to_string()).or_default()
                .insert(metric.to_string(), serde_json::json!(v));
        }
        serde_json::json!({
            "uptime_s": self.started.elapsed().as_secs(),
            "components": grouped,
        })
    }
}
