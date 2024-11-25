import save_centroid as cent
import apply_vector as vec
from utils_new import create_directory, load_data_df, model_to_str, save_rot_npz, get_all_angles
from datetime import datetime
import numpy as np
import os


def divide_dataframe_txt(df, txt_directory):
    with open(txt_directory) as file:
        lines = file.readlines()
    
    test_people = []
    train_people = []

    current_mode = None

    for line in lines:
        line = line.strip()
        if line.lower() == 'test':
            current_mode = 'test'
        elif line.lower() == 'train':
            current_mode = 'train'
        elif line:
            if current_mode == 'test':
                test_people.append(line)
            elif current_mode == 'train':
                train_people.append(line)

    df_test = df[df['person'].isin(test_people)]
    df_train = df[df['person'].isin(train_people)]
    string_len = len(train_people) + len(test_people)
    string = ''
    if string_len < 6:
        string = string + 'train'
        for person in train_people:
            string = string + f"_{str(person)}"
        string = string + '_test'
        for person in test_people:
            string = string + f"_{str(person)}"
    else:
        print("Too many people to put in filename.")
        string = f"train{str(len(train_people))}_test{str(len(test_people))}"
    return df_test, df_train, string

def divide_dataframe(df, test_num, train_num):
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    grouped = shuffled_df.groupby('person')
    unique_persons = list(grouped.groups.keys())
    altogether = test_num + train_num
    if altogether <= len(unique_persons):
        test_persons  = unique_persons[:test_num]
        train_persons = unique_persons[test_num:altogether]

        df_test = df[df['person'].isin(test_persons)]
        df_train = df[df['person'].isin(train_persons)]
    else:
        raise ValueError(f"Test and train people together should be less then {len(unique_persons)}!")

    return df_test, df_train


def apply_model(model_dir, in_directory, args):
    if os.path.isfile(model_dir):
        if os.path.isfile(in_directory):
            model_path = model_dir
            
            df = load_data_df(in_directory)
            if args.method == 'rotation':
                data_dir = save_rot_npz(df, args.out_dir1, in_directory)
                df = load_data_df(data_dir)

            if args.choose == 'normal':
                df_test, df_train = divide_dataframe(df, args.test, args.train)
            elif args.choose == 'txt':
                df_test, df_train, string = divide_dataframe_txt(df, args.txt)
            else:
                raise ValueError("Wrong input on args.choose!")



        centroids = cent.centre_data_people(df_train)
        centroids['centred_embedding'] = centroids['centred_embedding'].apply(lambda x: np.array(x))

        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        fileDirectory = os.path.join(args.out_dir1, f"{current_date}")
        create_directory(fileDirectory)

        cent.save_centroids(centroids, fileDirectory)    
        decoder_model, encoder_type = vec.init_decoder(model_path)
        
        if args.big:
            if args.choose == 'normal':
                dirname = f"{current_date}_train{str(args.train)}_test{str(args.test)}_{model_to_str(model_path)}"
            elif args.choose == 'txt':
                dirname = f"{current_date}_txt_{string}_{model_to_str(model_path)}"
            else:
                raise ValueError("Wrong input on args.choose!")
            out_dir = create_directory(os.path.join(args.out_dir2, dirname))
            
            angles = get_all_angles()
            for ang in angles:
                if args.goal == 'neut':
                    matrix_dir, out_directory = vec.init_file_matrix(model_path, out_dir, fileDirectory, args.base, ang)
                    vec.tranform_and_restore(decoder_model, encoder_type, df_test, matrix_dir, model_path, out_directory, args.method, args.alpha, args.base)

                elif args.goal == 'pos':
                    matrix_dir, out_directory = vec.init_file_matrix(model_path, out_dir, fileDirectory, ang, args.base)
                    vec.tranform_and_restore(decoder_model, encoder_type, df_test, matrix_dir, model_path, out_directory, args.method, args.alpha, args.pos)
        else:
            matrix_dir, out_directory = vec.init_file_matrix(model_path, out_dir, fileDirectory, args.base, args.pos)
            vec.tranform_and_restore(decoder_model, encoder_type, df_test, matrix_dir, model_path, out_directory, args.method, args.alpha, args.base)


def convert_restore(model_dir, in_directory, args):
    if os.path.isfile(model_dir):
        if os.path.isfile(in_directory):
            model_path = model_dir
            
            df = load_data_df(in_directory)
            if args.method == 'rotation':
                data_dir = save_rot_npz(df, args.out_dir1, in_directory)
                df = load_data_df(data_dir)

            if args.choose == 'normal':
                df_test, df_train = divide_dataframe(df, args.test, args.train)
            elif args.choose == 'txt':
                df_test, df_train, string = divide_dataframe_txt(df, args.txt)
            else:
                raise ValueError("Wrong input on args.choose!")


        decoder_model, encoder_type = vec.init_decoder(model_path)
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.choose == 'normal':
            dirname = f"{current_date}_conv_res_test{str(args.test)}_{model_to_str(model_path)}"
        elif args.choose == 'txt':
            dirname = f"{current_date}_conv_res_txt_{string}_{model_to_str(model_path)}"
        else:
            raise ValueError("Wrong input on args.choose!")
        
        out_dir = create_directory(os.path.join(args.out_dir2, dirname))

        angles = get_all_angles()
        for ang in angles:
            vec.restore(df_test, decoder_model, encoder_type, model_path, out_dir, ang, args.method)
        print(f"Images saved in directory: {out_dir}")  