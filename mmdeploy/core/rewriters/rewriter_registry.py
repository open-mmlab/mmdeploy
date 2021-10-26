from typing import Callable

from mmdeploy.utils.constants import Backend


class RewriterRegistry:
    """A registry that recoreds rewrite objects.

    Logically this class is a two-dimensional table which maintains an object
    list for each backend. The records can be inserted to this table through
    register.

    Members:
        _rewrite_records (Dict[Backend, Dict[str, Dict]]): A data structure
            which records the register message in a specific backend.

    Example:
        >>> FUNCTION_REGISTRY = RewriterRegistry()
        >>> @FUNCTION_REGISTRY.register_object(backend="default")
        >>> def add():
        >>>     return a + b
        >>> records = FUNCTION_REGISTRY.get_record("default")
    """

    # TODO: replace backend string with "Backend" constant
    def __init__(self):
        self._rewrite_records = dict()
        self.add_backend(Backend.DEFAULT.value)

    def _check_backend(self, backend: str):
        """Check if a backend has been supported."""
        if backend not in self._rewrite_records:
            raise Exception('Backend is not supported by registry.')

    def add_backend(self, backend: str):
        """Add a backend dictionary."""
        if backend not in self._rewrite_records:
            self._rewrite_records[backend] = dict()

    def get_records(self, backend: str) -> dict:
        """Get all registered records in record table."""
        self._check_backend(backend)
        records = self._rewrite_records[Backend.DEFAULT.value].copy()
        if backend != Backend.DEFAULT.value:
            records.update(self._rewrite_records[backend])
        return records

    def _register(self, name: str, backend: str, **kwargs):
        """The implementation of register."""
        self._check_backend(backend)
        self._rewrite_records[backend][name] = kwargs

    def register_object(self, name: str, backend: str, **kwargs) -> Callable:
        """The decorator to register an object."""
        self._check_backend(backend)

        def decorator(object):
            self._register(name, backend, _object=object, **kwargs)
            return object

        return decorator
