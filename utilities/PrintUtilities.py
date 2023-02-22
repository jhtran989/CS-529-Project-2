# FIXME: to print inside data structures (e.g., list), need __repr__ instead of __str__
def auto_str(cls):
    # def __str__(self):
    #     return '%s(%s)' % (
    #         type(self).__name__,
    #         ', '.join('%s=%s\n' % item for item in vars(self).items())
    #     )
    def __repr__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s\n' % item for item in vars(self).items())
        )
    cls.__repr__ = __repr__
    return cls