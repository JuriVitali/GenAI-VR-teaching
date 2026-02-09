import json
import os
import shutil
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger()

def merge_and_cleanup_session_logs(session_id: str, frontend_base_path: str, backend_log_path: str, output_path: str):
    merged_events = []
    
    frontend_session_dir = os.path.join(frontend_base_path, session_id)
    frontend_file_path = os.path.join(frontend_session_dir, "events.jsonl")

    # --- LETTURA LOG FRONTEND ---
    try:
        if os.path.exists(frontend_file_path):
            with open(frontend_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        # Qui sta la differenza: La riga è GIÀ l'evento
                        evt = json.loads(line)
                        
                        # Parsing Timestamp
                        ts_ms = evt.get("ts_unix_ms", 0)
                        dt_object = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                        iso_timestamp = dt_object.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        
                        unified_log = {
                            "timestamp": iso_timestamp,
                            "source": "frontend",
                            "session_id": session_id,
                            "event": evt.get("event_name"),
                            "frontend_seq": evt.get("seq"),
                            "payload": evt.get("payload_json"),
                            # User ID e Condition sono presenti in ogni riga nel tuo file
                            "user_id": evt.get("user_id"),
                            "condition": evt.get("condition")
                        }
                        merged_events.append(unified_log)

                    except json.JSONDecodeError:
                        print(f"[MERGER ERROR] JSON Error on line {line_num}")
                        continue
            
            print(f"[MERGER DEBUG] Frontend events loaded: {len(merged_events)}")

    except Exception as e:
        logger.error(f"frontend_read_error: {e}")
        print(f"[MERGER ERROR] Frontend read exception: {e}")


    # --- LETTURA LOG BACKEND ---
    backend_events_count = 0
    try:
        if os.path.exists(backend_log_path):
            with open(backend_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if session_id not in line:
                        continue
                    
                    try:
                        log_entry = json.loads(line)
                        if log_entry.get("session_id") == session_id:
                            log_entry["source"] = "backend"
                            merged_events.append(log_entry)
                            backend_events_count += 1
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"backend_read_error: {e}")
        print(f"[MERGER ERROR] Backend read exception: {e}")

    print(f"[MERGER DEBUG] Backend events loaded: {backend_events_count}")

    # --- ORDINAMENTO E SALVATAGGIO ---
    merged_events.sort(key=lambda x: x.get("timestamp", ""))

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for entry in merged_events:
                f_out.write(json.dumps(entry) + "\n")
        
        logger.info("merge_success", output=output_path)
        
        #if os.path.exists(frontend_session_dir):
         #    shutil.rmtree(frontend_session_dir)
          #   logger.info("cleanup_success", deleted_dir=frontend_session_dir)
        
        #if os.path.exists(backend_log_path):
         #   try:
          #      os.remove(backend_log_path)
           #     print(f"[MERGER CLEANUP] Deleted backend log: {backend_log_path}")

            #except OSError as e:
             #   print(f"[MERGER ERROR] Could not delete backend log (file locked?): {e}")
            
        print(f"[MERGER SUCCESS] LOG MERGE COMPLETED. SESSION: {session_id} | SAVED TO: {output_path}")
        return True

    except Exception as e:
        logger.error(f"merge_write_error: {e}")
        print(f"[MERGER FAILED] Error writing output: {e}")
        return False