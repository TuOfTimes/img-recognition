import os

class OutputHandler():

    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.test_scores = self.get_test_scores()

    def get_test_scores(self):
        test_scores = {}
        test_folder = self.folder_path + "TestScores/"
        files = os.listdir(test_folder)

        for file_path in files:
            if file_path != "placeholder.txt":
                data =[]
                with open(test_folder+file_path,"r") as f:
                    data = f.readlines()
                score = float(data[-1].split(":")[-1])

                test_scores[file_path.strip(".txt")] = score

        return test_scores



if __name__ == "__main__":
    oh = OutputHandler("Outputs/")
    print("10 Highest Test Scores")

    count = 0
    for key,val in sorted(oh.test_scores.items(), key=lambda kv: kv[1],reverse=True):
        print("Configuration: {}  - Test Accuracy: {} ".format(key,val))