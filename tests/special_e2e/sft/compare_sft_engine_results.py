import os


def get_result(file):
    file = os.path.expanduser(file)
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "step:" in line:
                result = line.split(" - ")

                loss_result = result[1]
                grad_norm = result[2]

                from IPython import embed
                embed()




if __name__ == "__main__":
    get_result("~/verl/test/log/golden.log")