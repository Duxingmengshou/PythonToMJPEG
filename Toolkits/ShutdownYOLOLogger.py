import logging

'''
用于关闭YOLO默认的控制台输出（类别之类的识别信息）
'''
def ShutdownYOLOLogger():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.disabled = True
