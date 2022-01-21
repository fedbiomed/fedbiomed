class RepositoryCache(object):

    file_sizes = {}

    @classmethod
    def clear(cls, path):
        if path in cls.file_sizes:
            del cls.file_sizes[path]

    @classmethod
    def clear_all(cls):
        cls.file_sizes = {}
