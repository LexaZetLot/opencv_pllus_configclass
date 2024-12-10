def Bean(nameBean=None, fabricMethod = 'singleton'):
    def onDecorator(func):
        def onCall(*args, **kwargs):
            func.__dict__['BeanSetting'] = [nameBean, fabricMethod]
            return func, args, kwargs
        return onCall
    return onDecorator
