def Autowired():
    def onDecorator(func):
        func.__dict__['Autowired'] = True
        return func
    return onDecorator