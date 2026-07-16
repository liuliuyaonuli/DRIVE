import traceback


def build_error_status(error: BaseException) -> dict:
    error_text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    return {
        "done": True,
        "reward": 0,
        "success": 0.0,
        "num_actions": 0,
        "error": error_text,
    }
