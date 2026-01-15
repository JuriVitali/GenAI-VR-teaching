import time
import functools
import structlog

logger = structlog.get_logger()

def log_event(action_name, result_mapper=None):
    """
    action_name: The base name for the event (e.g., 'audio_transcription')
    result_mapper: A function that takes the return value and returns a dict to log.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log Start
            logger.info(f"{action_name}_started")
            
            start_time = time.perf_counter()
            try:
                # Run the actual function (returns original type, e.g., string)
                real_result = func(*args, **kwargs)
                
                duration = time.perf_counter() - start_time
                
                # Prepare Log Data
                log_payload = {"duration": duration}
                
                # Map the result if a mapper is provided
                if result_mapper:
                    # Transform the string/object into the dict keys you want
                    custom_data = result_mapper(real_result)
                    if isinstance(custom_data, dict):
                        log_payload.update(custom_data)

                # Log Finish with the mapped data
                logger.info(f"{action_name}_finished", **log_payload)
                
                # Return the ORIGINAL result (unmodified)
                return real_result

            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(
                    f"{action_name}_failed", 
                    error=str(e), 
                    duration=duration
                )
                raise
        return wrapper
    return decorator