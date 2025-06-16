### V1
```json
{
  "schema_version": "1.0",                // fixed schema version
  "status": "ok",                         // or "error"
  "messages": [
    {
      "turn_id": 7,
      "role": "assistant",
      "content": "Sure, here's a concise example in C...",
      "timestamp": 1718611200.0
    }
  ],
  "intent": "provide_code_example",       // what the assistant is doing
  "entities": {
    "language": "C",
    "task": "vowel counter"
  },
  "next_action": {
    "name": "none",
    "args": {}
  },
  "memory_read": [
    "User likes concise examples"
  ],
  "memory_write": [],
  "warnings": [],
  "retrieval": {
    "method": "semantic",
    "query": "vowel counter",
    "results": [
      { "text": "Use tolower() for case insensitivity", "score": 0.22 },
      { "text": "Loop over strlen(str)",               "score": 0.35 }
    ]
  }
}
```