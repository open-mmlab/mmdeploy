# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
import shutil

import torch
from pyppeteer import launch
from torchvision.models import resnet18

from mmdeploy.core import FUNCTION_REWRITER, RewriterContext, patch_model
from mmdeploy.utils import get_root_logger


@FUNCTION_REWRITER.register_rewriter(
    func_name='torchvision.models.ResNet._forward_impl')
def forward_of_resnet(ctx, self, x):
    """Rewrite the forward implementation of resnet.

    Early return the feature map after two down-sampling steps.
    """
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    return x


def rewrite_resnet18(original_path: str, rewritten_path: str):
    # prepare inputs and original model
    inputs = torch.rand(1, 3, 224, 224)
    original_model = resnet18(pretrained=False)

    # export original model
    torch.onnx.export(original_model, inputs, original_path)

    # patch model
    patched_model = patch_model(original_model, cfg={}, backend='default')

    # export rewritten onnx under a rewriter context manager
    with RewriterContext(cfg={}, backend='default'), torch.no_grad():
        torch.onnx.export(patched_model, inputs, rewritten_path)


def screen_size():
    """Get windows size through tkinter."""
    import tkinter
    tk = tkinter.Tk()
    width = tk.winfo_screenwidth()
    height = tk.winfo_screenheight()
    tk.quit()
    return width, height


async def visualize(original_path: str, rewritten_path: str):
    # launch a web browser
    browser = await launch(headless=False, args=['--start-maximized'])
    # create two new pages
    page2 = await browser.newPage()
    page1 = await browser.newPage()
    # go to netron.app
    width, height = screen_size()
    await page1.setViewport({'width': width, 'height': height})
    await page2.setViewport({'width': width, 'height': height})
    await page1.goto('https://netron.app/')
    await page2.goto('https://netron.app/')
    await asyncio.sleep(2)

    # open local two onnx files
    mupinput1 = await page1.querySelector("input[type='file']")
    mupinput2 = await page2.querySelector("input[type='file']")
    await mupinput1.uploadFile(original_file_path)
    await mupinput2.uploadFile(rewritten_file_path)
    await asyncio.sleep(4)
    for _ in range(6):
        await page1.click('#zoom-out-button')
        await asyncio.sleep(0.3)
    await asyncio.sleep(1)
    await page1.screenshot({'path': original_path.replace('.onnx', '.png')},
                           clip={
                               'x': width / 4,
                               'y': 0,
                               'width': width / 2,
                               'height': height
                           })
    await page2.screenshot({'path': rewritten_path.replace('.onnx', '.png')},
                           clip={
                               'x': width / 4,
                               'y': 0,
                               'width': width / 2,
                               'height': height
                           })
    await browser.close()


if __name__ == '__main__':
    tmp_dir = os.getcwd() + '/tmp'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    original_file_path = os.path.join(tmp_dir, 'original.onnx')
    rewritten_file_path = os.path.join(tmp_dir, 'rewritten.onnx')
    logger = get_root_logger()
    logger.info('Generating resnet18 and its rewritten model...')
    rewrite_resnet18(original_file_path, rewritten_file_path)

    logger.info('Visualizing models through netron...')
    asyncio.get_event_loop().run_until_complete(
        visualize(original_file_path, rewritten_file_path))
    import mmcv
    image1 = mmcv.imread(original_file_path.replace('.onnx', '.png'))
    image2 = mmcv.imread(rewritten_file_path.replace('.onnx', '.png'))
    mmcv.imshow(image1, win_name='original')
    mmcv.imshow(image2, win_name='rewritten')
    shutil.rmtree(tmp_dir)
