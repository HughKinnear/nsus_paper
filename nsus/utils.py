
def verbose(method):
    def wrapper(self, *args, **kwargs):
        if self.verbose:
            return method(self, *args, **kwargs)
    return wrapper
