import functools
import hashlib
import inspect
import json
import os


def persistent_cache(cache_file=None):
    """
    Decorator that creates a persistent cache for function calls.

    Args:
        cache_file (str, optional): Path to the cache file. If None, uses the function name.

    Returns:
        function: Decorated function with persistent caching.
    """

    def decorator(func):
        # Get the file path for the cache
        if cache_file is None:
            # Default to function name in current working directory if module path not available
            try:
                module_path = inspect.getmodule(func).__file__
                module_dir = os.path.dirname(os.path.abspath(module_path))
            except AttributeError:
                module_dir = os.getcwd()
            cache_path = os.path.join(module_dir, f"{func.__name__}_cache.json")
        else:
            cache_path = cache_file

        # Load existing cache if it exists
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                cache = {}
        else:
            cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hash of the arguments to use as a cache key
            # We need to handle non-hashable arguments (like lists and dicts)
            key_parts = []

            # Add function name to ensure different functions don't share cache keys
            key_parts.append(func.__name__)

            # Process positional arguments
            for arg in args:
                if isinstance(arg, (list, dict, set)):
                    # Convert to a string representation for hashing
                    key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest())
                else:
                    key_parts.append(str(arg))
            # Process keyword arguments (sorted for consistency)
            for k in sorted(kwargs.keys()):
                v = kwargs[k]
                if isinstance(v, (list, dict, set)):
                    key_parts.append(f"{k}={hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest()}")
                else:
                    key_parts.append(f"{k}={v}")

            # Create the final cache key
            cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()

            # Check if result is in cache
            if cache_key in cache:
                print(f"Cache hit for {func.__name__}! Returning cached result.")
                return cache[cache_key]

            # If not in cache, call the function and store the result
            result = func(*args, **kwargs)

            # Try to make result JSON serializable
            try:
                # Test if the result is JSON serializable
                json.dumps(result)
                cache[cache_key] = result

                # Save the updated cache
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=2)
            except (TypeError, OverflowError):
                print(f"Warning: Result from {func.__name__} is not JSON serializable. Not caching.")

            return result

        return wrapper

    return decorator
