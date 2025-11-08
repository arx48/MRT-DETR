# https://blog.csdn.net/qq_40672115/article/details/134356250?ops_request_misc=&request_id=&biz_id=102&utm_term=RT-DETR&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-134356250.142%5ev100%5epc_search_result_base5&spm=1018.2226.3001.4187

import torch
import onnxruntime as ort
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    # print(onnx.helper.printable_graph(mm.graph))

    im = Image.open('test_pic/000000107339.jpg').convert('RGB')
    im = im.resize((640, 640))
    im_data = ToTensor()(im)[None]
    print(im_data.shape)

    size = torch.tensor([[640, 640]])
    sess = ort.InferenceSession("model.onnx")
    output = sess.run(
        # output_names=['labels', 'boxes', 'scores'],
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
    )

    # print(type(output))
    # print([out.shape for out in output])

    labels, boxes, scores = output

    draw = ImageDraw.Draw(im)
    thrh = 0.6

    for i in range(im_data.shape[0]):

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        print(i, sum(scr > thrh))

        for b in box:
            draw.rectangle(list(b), outline='red', width=3)
            draw.text((b[0], b[1]), text=str(lab[i]), fill='blue', )
            # draw.text((b[0], b[1]+2), text=str(scr[i]), fill='green', )

    im.save('test_pic/test000000107339.jpg')
