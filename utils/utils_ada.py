import models.AdaFace.net as net
import torch
import os
from models.AdaFace.face_alignment import align
import numpy as np

adaface_models = {
    'ir_101':"/home/rokp/test/models/AdaFace/AdaFace/pretrained/adaface_ir101_webface12m.ckpt",
}
def load_pretrained_model(architecture='ir_101'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor



if __name__ == '__main__':

    model = load_pretrained_model('ir_101')
    feature, norm = model(torch.randn(2,3,112,112))
    embeddings = []

    test_image_path = '/home/rokp/test/models/AdaFace/AdaFace/face_alignment/test_images'
    features = []
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = align.get_aligned_face(path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        embedding_np = feature.detach().numpy()
        embeddings.append(embedding_np)

        print(f"Embedding for {fname}: {embedding_np.shape}")
        features.append(feature)

    embeddings_np = np.vstack(embeddings)
    print(f"All embeddings as NumPy array: {embeddings_np}")

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(similarity_scores)