import inspect
from types import NoneType


class AnnotationConfigApplicationContext(object):
    def __init__(self, ConfigClass):
        self.__ConfigClass = ConfigClass
        self.__listBean = []
        self.__beanSingleton = {}
        self.__beanPrototype = {}
        self.__classCreate = ConfigClass()

        self._searchBeanAnnotation(self.__ConfigClass, 'Bean', self.__listBean)



    def getBean(self, nameBean):
        listAutowired = []
        listArg = []
        listParam = []

        self.__generateBeanReturn(nameBean)

        if nameBean in self.__beanSingleton.keys():
            resultFabric = self.__beanSingleton[nameBean]
            AnnotationConfigApplicationContext._searchAutowiredAnnotation(resultFabric,  listAutowired)
            for autowired in listAutowired:
                self.__generateBeanReturn(autowired)
                listArg = AnnotationConfigApplicationContext._genericArgsFunc(resultFabric, autowired)
                for args in listArg:
                    if args in self.__beanSingleton.keys():
                        listParam.append(self.__beanSingleton[args])
                    elif args in self.__beanPrototype.keys():
                        listParam.append(self.__beanPrototype[args][0](self.__beanPrototype[args][1],
                                                                       *self.__beanPrototype[args][2],
                                                                       **self.__beanPrototype[args][3]))
                resultFabric.__class__.__dict__[autowired](resultFabric, *listParam)
            return resultFabric
        else:
            resultFabric = self.__beanSingleton[nameBean]
            AnnotationConfigApplicationContext._searchAutowiredAnnotation(resultFabric,  listAutowired)
            for autowired in listAutowired:
                self.__generateBeanReturn(autowired)
                listArg = AnnotationConfigApplicationContext._genericArgsFunc(resultFabric, autowired)
                for args in listArg:
                    if args in self.__beanSingleton.keys():
                        listParam.append(self.__beanSingleton[args])
                    elif args in self.__beanPrototype.keys():
                        listParam.append(self.__beanPrototype[args][0](self.__beanPrototype[args][1],
                                                                       *self.__beanPrototype[args][2],
                                                                       **self.__beanPrototype[args][3]))
                resultFabric.__class__.__dict__[autowired](resultFabric, *listParam)
            return resultFabric

    @staticmethod
    def _searchBeanAnnotation(clas, annotationName, listBean):
        for name in [name for name in clas.__dict__ if not name.startswith('__')]:
            if str(clas.__dict__[name]).count(annotationName):
                func, _, _, = clas.__dict__[name]()

                if func.__dict__['BeanSetting'][0] is not None:
                    setattr(clas, func.__dict__['BeanSetting'][0], clas.__dict__[name])
                    listBean.append(func.__dict__['BeanSetting'][0])
                else:
                    listBean.append(name)
                del func

    @staticmethod
    def _searchAutowiredAnnotation(clas, listSave):
        for dic in clas.__class__.__dict__:
            if not dic.startswith('__') and not dic.startswith('_abc'):
                string = str(clas.__class__.__dict__[dic].__dict__)
                if string.find('Autowired') > 0:
                    if string[string.find('Autowired') + 12: string.find('Autowired') + 16] == 'True':
                        listSave.append(dic)

    @staticmethod
    def _genericArgsFunc(clas, autowired):
        argsName = inspect.signature(clas.__class__.__dict__[autowired])
        argsName = argsName.parameters.keys()
        argsName = str(argsName)
        argsName = argsName[13:]
        argsName = argsName[:-3]
        listArgsName = argsName.split("', '")
        listRes = []
        for arg in listArgsName:
            if arg != 'self':
                listRes.append('get' + arg[0].upper() + arg[1:])
        return listRes

    def __generateBeanReturn(self, nameBean):
        if nameBean in self.__listBean:
            func, argv, kwarg = self.__ConfigClass.__dict__[nameBean]()
            if func.__dict__['BeanSetting'][0] is None:
                if func.__dict__['BeanSetting'][1] == 'singleton':
                    self.__beanSingleton[nameBean] = func(self.__classCreate, *argv, **kwarg)
                elif func.__dict__['BeanSetting'][1] == 'prototype':
                    self.__beanPrototype[nameBean] = [func, self.__classCreate, argv, kwarg]
            elif func.__dict__['BeanSetting'][0] is not None:
                if func.__dict__['BeanSetting'][1] == 'singleton':
                    self.__beanSingleton[func.__dict__['BeanSetting'][0]] = func(self.__classCreate, *argv, **kwarg)
                elif func.__dict__['BeanSetting'][1] == 'prototype':
                    self.__beanPrototype[func.__dict__['BeanSetting'][0]] = [func, self.__classCreate, argv, kwarg]
        else:
            raise NoneType
