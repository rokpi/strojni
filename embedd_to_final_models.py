import argparse
import os
from utils_embed_to_final import apply_model, convert_restore

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', default=r'/home/rokp/test/models', type=str, help='Path to the models')
    parser.add_argument('--method', default= 'centroid', type=str, help='Whether the vector is rotation or centroid.')
    parser.add_argument('--search', default=r'arcface-resnet', type=str, 
    help='What to examine. (xxx-xxx) or (retinaface, mtcnn, raw), can be combination of both (eg. mtcnn/resnet-resnet)')

    parser.add_argument('--convert_restore', default=True, type=bool, help='Use rotation space and restore.')
    parser.add_argument('--big', default=True, type=bool, help='Make it big')
    parser.add_argument('--normal', default='txt', type=str, help='Way to choose the embeddings(txt or normal)')
    parser.add_argument('--train', default= 1, type=int, help='Number of peoples embeddings to take for training(if --choose == normal).')
    parser.add_argument('--test', default= 1, type=int, help='Number of peoples embeddings to take for testing(if --choose == normal).')
    parser.add_argument('--txt', default='/home/rokp/test/launch_embedd_250.txt', type=str, help='File with test and train people numbers.')

    #create centroid
    parser.add_argument('--out_dir1', default=r'/home/rokp/test/centroid', type=str, 
    help='Output directory where the embeddings will be saved.')

    #apply vector 
    
    parser.add_argument('--base', default= 0, type=int, help='Viewpoint of images to tranform to.')
    parser.add_argument('--goal', default= 'neut', type=str, help='Whether the base is the neut or pos centroid.')
    parser.add_argument('--out_dir2', default='/home/rokp/test/transformed_embeddings', type=str, help='Output directory.')
    parser.add_argument('--alpha', default= 3.5, type=int, help='Value of alpha.')
    
    args = parser.parse_args()
    return args


def check_correct(model_str):
    preprocess = ['mtcnn', 'retinaface', 'raw']
    models = ['vgg-resnet', 'vgg-vgg', 'resnet-vgg', 'resnet-resnet', 'arcface-resnet']
    result = 'False'
    prep = ''
    model = ''
    if '/' in model_str:
        split = model_str.split('/')
        prep = split[0]
        model = split[1]
        if prep in preprocess and model in models:
            result = 'prep_model'
        else:
            raise NameError(f"Wrong input of args.search: {model_str}")

    elif model_str in preprocess:
        result = 'prep'
        prep = model_str
    elif model_str in models:
        result = 'model'
        model = model_str
    else:
        raise NameError(f"wrong input of args.models: {model_str}")
    return result, prep, model


def main(args):
    models_directory = args.models_dir
    decision, prep, model = check_correct(args.search)
    if os.path.isdir(models_directory):

        if decision in ['prep','prep_model']:
            models_directory_list = [f for f in os.listdir(models_directory) if f == prep]
        else:
            models_directory_list = os.listdir(models_directory)
        for preproc in models_directory_list:    
            preproc_dir = os.path.join(models_directory, preproc)
            if os.path.isdir(preproc_dir):

                if decision in ['model', 'prep_model']:
                    model_subdirs = [f for f in os.listdir(preproc_dir) if f == model]
                else:
                    model_subdirs = os.listdir(preproc_dir)    
                for model_subdir in model_subdirs:
                    dirs = os.path.join(preproc_dir, model_subdir)
                    model_dirs = [f for f in os.listdir(dirs) if f.endswith('.hdf5')]
                    in_dir = [f for f in os.listdir(dirs) if f.endswith('.npz')][0]
                    in_directory = os.path.join(dirs, in_dir)
                    for mod_dir in model_dirs:
                        model_dir = os.path.join(dirs, mod_dir)
                        if args.convert_restore:
                            convert_restore(model_dir, in_directory, args)
                        else:
                            apply_model(model_dir, in_directory, args)
                    



if __name__ == '__main__':
  args = argparser()
  main(args)