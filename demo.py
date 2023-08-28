import matplotlib.pyplot as plt
import torch
import os
from safetensors.torch import load_model, save_model
import numpy as np
import cv2
from lightglue import LightGlue, SuperPoint


def process_image(image, max_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    scale = max(h, w) / max_size
    h_new, w_new = int(round(h / scale)), int(round(w / scale))
    return cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_AREA), scale


with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = SuperPoint(
        max_num_keypoints=1024,
        nms_radius=4,
        detection_threshold=0.0005
    )
    if not os.path.exists('weights/superpoint.safetensor'):
        print("Converting superpoint to safetensors")
        extractor.load_state_dict(torch.load("weights/superpoint_v1.pth"))
        save_model(extractor, 'weights/superpoint.safetensor')
    else:
        load_model(extractor, 'weights/superpoint.safetensor')
    extractor = extractor.to(device).eval()

    matcher = LightGlue(
        n_layers=9,
        flash=False,
        mp=False,
        depth_confidence=-1,
        width_confidence=-1,
        filter_threshold=0.1
    )
    if not os.path.exists('weights/lightglue.safetensor'):
        print("Converting lightglue to safetensors")
        matcher.load_state_dict(torch.load(
            "weights/superpoint_lightglue.pth"), strict=False)
        save_model(matcher, 'weights/lightglue.safetensor')
    else:
        load_model(matcher, 'weights/lightglue.safetensor', strict=False)
    matcher = matcher.to(device).eval()

    image0 = cv2.imread("assets/sacre_coeur1.jpg")
    image1 = cv2.imread("assets/sacre_coeur2.jpg")

    gray0, scales0 = process_image(image0, 512)
    gray1, scales1 = process_image(image1, 512)
    tensor0 = torch.from_numpy(gray0[None][None]).to(device).float() / 255.0
    tensor1 = torch.from_numpy(gray1[None][None]).to(device).float() / 255.0
    size0 = np.array([gray0.shape[1], gray0.shape[0]])
    size1 = np.array([gray1.shape[1], gray1.shape[0]])
    size0 = torch.from_numpy(size0).to(device).float()
    size1 = torch.from_numpy(size1).to(device).float()

    if not os.path.exists('weights/superpoint.ptc'):
        print("Saving superpoint traced model")
        traced_extractor = torch.jit.trace(extractor, tensor0)
        torch.jit.save(traced_extractor, 'weights/superpoint.ptc')
    else:
        traced_extractor = torch.jit.load('weights/superpoint.ptc')
    kpts0, scores0, desc0 = extractor.forward(tensor0)
    kpts1, scores1, desc1 = extractor.forward(tensor1)
    if not os.path.exists('weights/lightglue.ptc'):
        print("Saving lightglue traced model")
        traced_matcher = torch.jit.trace(
            matcher, (size0, size1, kpts0, kpts1, desc0, desc1))
        torch.jit.save(traced_matcher, 'weights/lightglue.ptc')
    else:
        traced_matcher = torch.jit.load('weights/lightglue.ptc')
    result = matcher.forward(size0, size1, kpts0, kpts1, desc0, desc1)

    def match():
        gray0, scales0 = process_image(image0, 512)
        gray1, scales1 = process_image(image1, 512)
        tensor0 = torch.from_numpy(
            gray0[None][None]).to(device).float() / 255.0
        tensor1 = torch.from_numpy(
            gray1[None][None]).to(device).float() / 255.0
        size0 = np.array([gray0.shape[1], gray0.shape[0]])
        size1 = np.array([gray1.shape[1], gray1.shape[0]])
        size0 = torch.from_numpy(size0).to(device).float()
        size1 = torch.from_numpy(size1).to(device).float()
        kpts0, _, desc0 = traced_extractor.forward(tensor0)
        kpts1, _, desc1 = traced_extractor.forward(tensor1)
        result = traced_matcher.forward(
            size0, size1, kpts0, kpts1, desc0, desc1)
        mkpts0, mkpts1, matches, mscores = result
        kpts0 = kpts0.cpu().numpy() * scales0
        kpts1 = kpts1.cpu().numpy() * scales1
        mkpts0 = mkpts0.cpu().numpy() * scales0
        mkpts1 = mkpts1.cpu().numpy() * scales1
        matches = matches.cpu().numpy()
        mscores = mscores.cpu().numpy()
        return kpts0, kpts1, mkpts0, mkpts1, matches, mscores

    kpts0, kpts1, mkpts0, mkpts1, matches, mscores = match()

    import timeit
    print(timeit.timeit(match, number=10) / 10)
    print(timeit.timeit(match, number=10) / 10)

    print("Keypoints0:", kpts0.shape)
    print("Keypoints1:", kpts1.shape)
    print("Min score0:", scores0.min())
    print("Min score1:", scores1.min())
    print("Max score0:", scores0.max())
    print("Max score1:", scores1.max())

    print("Matches0:", mkpts0.shape)
    print("Matches1:", mkpts1.shape)
    print("Min score:", mscores.min())
    print("Max score:", mscores.max())

    plt.subplot(121)
    plt.imshow(image0[:, :, ::-1])
    plt.scatter(kpts0[0, :, 0],
                kpts0[0, :, 1], s=1, c='r')
    plt.scatter(mkpts0[0, :, 0],
                mkpts0[0, :, 1], s=1, c='b')
    plt.subplot(122)
    plt.imshow(image1[:, :, ::-1])
    plt.scatter(kpts1[0, :, 0],
                kpts1[0, :, 1], s=1, c='r')
    plt.scatter(mkpts1[0, :, 0],
                mkpts1[0, :, 1], s=1, c='b')
    plt.show()
