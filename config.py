#-*-encoding:utf8-*-#
import configparser
import os
cwd = os.path.dirname(__file__)

"""读取配置文件-反复用的一部分代码段:
将配置文件里的字段转换成最有可能的类型（如果既不能转换成整数或小数，则保留为字符串）
"""
def config_from_sec(sec_list, cf, section_name):
    config_dict ={}
    for opt_name in sec_list:
        opt_value = cf.get(section_name, opt_name)
        #这一步的配置值都是字符串，要转类型
        try:
            config_dict[opt_name] = int(opt_value)
        except ValueError: #不是整数，试下小数
            try:
                config_dict[opt_name] = float(opt_value)
            except ValueError:  # 不是小数，那就还是字符串
                config_dict[opt_name] = opt_value
    return config_dict


"""读取训练配置文件"""
def load_config_file(configFile = "model.config"):
    global cwd
    configFile = os.path.join(cwd, configFile)
    cf = configparser.ConfigParser()
    cf.read(configFile)
    task_sec = cf.options("task")
    hyper_parameters_sec = cf.options("hyper_parameter")
    extra_parameters_sec = cf.options("extra_parameter")

    task_dict = config_from_sec(task_sec, cf, "task")
    hyper_parameters_dict = config_from_sec(hyper_parameters_sec, cf, "hyper_parameter")
    extra_parameters_dict = config_from_sec(extra_parameters_sec, cf, "extra_parameter")
    return task_dict, hyper_parameters_dict, extra_parameters_dict

"""脚本的命令行输入提示"""
def printUsage():
    print("usage: classify_xgboost_offline_train.py -f <configFileName> -a [test|train]")

if __name__ == "__main__":
    """这个主调用
    用途是读 构造好了的输入文件 ，然后训练模型/预测输出csv
    所谓 构造好了的输入文件，是经由csvHandling.py运行后产生的输入文件，记下他的时间前缀
    它存在的意义是为了不用反复跑csvHandling.py
    因为本身可以用同一组输入文件跑出很多个不同参数的模型，每次测的时候都从csvHandlling.py跑太慢了，输入文件是一样的（模型词典不变的情况下）
    直接在这里改参数，就可以跑一系列输入文件相同但参数不同的模型
    """
    import getopt, sys
    try:
        opts, args = getopt.getopt(sys.argv[1:],"a:f:",["action=","file="])
    except getopt.GetoptError:
        #参数错误
        printUsage()
        sys.exit(-1)

    action = "train"
    configFile = "train_models.config"
    for opt,arg in opts:
        if opt in ("-a", "--action"):
            action = arg
        elif opt in ("-f", "--file"):
            configFile = arg
    #init_train_model(action, configFile)
