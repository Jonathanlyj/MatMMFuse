import yaml
import training

def run_supervised():

    with open( "./config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    training.train_regular(
                        "cuda",
                        0,
                        "./data/",
                        config["Job"]['Training'],
                        config["Training"],
                        config["Models"]["CGCNN_demo"],
                        )


    training.predict(data_path= "./data/" ,
                     training_parameters = config['Training'] ,
                     model_parameters=config["Models"]["CGCNN_demo"], job_parameters= config["Job"]['Predict'])
