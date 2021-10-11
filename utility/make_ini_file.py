import configparser

def print_config(config: configparser.ConfigParser):
    for section in config.sections():
        print("=====Section=====")
        print(section)
        print("=================")
        for item in config[section]:
            print("{}: {}".format(item, config[section][item]))

def update_config_for_ger(input_path: str, output_path: str):
    config = configparser.ConfigParser()
    config.read(input_path)
    print_config(config)

    response = input_function()
    while response != "close":
        config[response[0]][response[1]] = response[2]
        response = input_function()
    with open(output_path, "w") as configfile:
        config.write(configfile)


def input_function():
    print("Enter 's' for saving, otherwise enter a section!")
    section = str(input())
    if section == "s":
        print("New ini-file gets saved now...")
        return "close"
    else:
        print("Enter a Key")
        key = str(input())
        print("Enter a new Value for that Key")
        value = str(input())
        return [section, key, value]


def main(input_path: str, output_path:str):
    update_config_for_ger(input_path, output_path)

if __name__ == "__main__":
    main("/home/stud_homes/s5935481/work3/data/configs/ptb.crf2o.dep.lstm.char.ini",
         "/home/stud_homes/s5935481/work3/data/configs/costum01.ini")

