import os
import argparse

from my_utils.utils_new import  create_directory, getAngle,get_all_filepaths
from tqdm import tqdm
from datetime import datetime
import shutil

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=r'/home/rokp/test/images_extract', type=str, help='Output directory where the embeddings will be saved.')
    parser.add_argument('--target', default='txt', type=str, help='What images to copy (txt or frontal(of all people))')    
    parser.add_argument('--inputs', default=r'/home/rokp/test/images/images_mtcnn', type=str, help='A file containing the names of test images, an input image or a directory containing the input images.') 
    parser.add_argument('--txt', default='/home/rokp/test/strojni/launch/launch_test_arcface.txt', type=str, help='Directory to text file.')     
    args = parser.parse_args()
    return args

def read_txt(txt_directory):
    with open(txt_directory) as file:
        lines = file.readlines()  
    people = []
    for line in lines:
        line = line.strip()
        if line:
            people.append(line)
    print(f"Persons: {people}.")
    return people


def check_img_requirements_frontal(img_name):
    if img_name[7:9] == '01':                       
        if img_name[14:16] in ['08']: #'07',
            face_viewpoint = img_name[10:13]
            if face_viewpoint in ['051']: #'130', '140', , '050', '041'
                return True
            
    return False

def check_img_requirements_person(img_name, people):
    person_new = img_name[0:3] 
    if person_new in people:
        if img_name[7:9] == '01':                       
            #if img_name[14:16] in ['08']: #'07',
                face_viewpoint = img_name[10:13]
                if face_viewpoint in ['130', '140', '051', '050', '041']: 
                    return True
            
    return False

def make_directories(input_dir, target, people_list):
    filtered_img_list = []
    seen_persons = set()
    num = 0
    num_of_img = 0
    for img in input_dir:    
        filename = os.path.basename(img)
        person_new = filename[0:3]  
        if person_new.isdigit():
            if target == 'frontal':
                requirements = check_img_requirements_frontal(filename)
            elif target == 'txt':
                requirements = check_img_requirements_person(filename, people_list)
            else:
                raise NameError(f"Wrong input of args --target: {target}")
            if requirements:
                if person_new not in seen_persons: #and num <= num_people:
                    num = num + 1
                    seen_persons.add(person_new)
                    filtered_img_list.append({'person': person_new, 'img_dir': filename}) #,'angle': getAngle(img)})
                    num_of_img = num_of_img + 1
                elif person_new in seen_persons:
                    filtered_img_list.append({'person': person_new, 'img_dir': filename}) #,'angle': getAngle(img)}) 
                    num_of_img = num_of_img + 1
                    '''if num_of_img >= expected_num_of_img:
                        print("All done with img acquiring...")
                        return filtered_img_list'''
                        
        else:
            raise ValueError("Unexpected person classification (first three letters of basename should be an integer)")  
    print(f"Not as many as expected. Num of imgs: {num_of_img}.")
    print(f"Num of different people: {len(seen_persons)}")
    return filtered_img_list

def main(args):

    #reads args.src, goes to directory and finds path/paths to file/files
    if os.path.isdir(args.inputs):

        in_directory = args.inputs
        people_list = None
        #makes the directory
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.target == 'frontal':
            dirname = f"{args.target}_{current_date}_imgs"
        elif args.target == 'txt':
            dirname = f"{args.target}_{current_date}_imgs"
        else:
            raise NameError(f"Wrong input of args --target: {args.target}")   
                 
        out_directory = create_directory(os.path.join(args.out_dir, dirname))

        img_list = get_all_filepaths(in_directory,'.jpg')

        if args.target == 'txt':
            people_list = read_txt(args.txt)
        filtered_img_list = make_directories(img_list, args.target, people_list)

        for item in tqdm(filtered_img_list, total = len(filtered_img_list)):
            img_dir = item['img_dir']
            file_path = os.path.join(in_directory, img_dir)
            if os.path.isfile(file_path):
                out_path = os.path.join(out_directory, img_dir)
                shutil.copy(file_path, out_path)

            else:
                print("File %s doesn't exist." % file_path)
        print(f"images saved in directory: {out_directory}")
    
if __name__ == '__main__':
  args = argparser()
  main(args)


    

