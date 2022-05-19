from fairseq.models.delight_transformes import DeLighTTransformerModel
# from mod


import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

# print(args)

delight_model = DeLighTTransformerModel.build_my_model(args, 35000,0,35000,0)



