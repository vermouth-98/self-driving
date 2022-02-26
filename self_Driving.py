from model import ModelPL
import torch
from torchvision import transforms as T
from screen.screen_recorder import ImageSequencer
from keyboard.getkeys import key_check
from screen.screen_recorder import ImageSequencer
import logging
import time
from tkinter import *
import numpy as np
import cv2
from utils import mse
from keyboard.inputsHandler import select_key
from keyboard.getkeys import id_to_key
import math

def main(
    width,
    height,
    full_screen,
    show_current_control,
    dtype = torch.float32,
    enable_evasion = False,
    evasion_score=1000
    ):
    device = torch.device("cuda:0")
    model = ModelPL.load_from_checkpoint("modelpl.ckpt")
    model.eval()
    model.to(dtype = torch.float32,device = device)
    show_what_ai_sees: bool = False
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    img_sequencer = ImageSequencer(
        width=width,
        height=height,
        full_screen=full_screen,
        get_controller_input=False,
        num_sequences=4,
        total_wait_secs=5,
    )
    if show_current_control:
        root = Tk()
        var = StringVar()
        var.set("Driving")
        text_label = Label(root, textvariable=var, fg="green", font=("Courier", 44))
        text_label.pack()
    else:
        root = None
        var = None
        text_label = None
    last_time: float = time.time()
    score: np.float = np.float(0)
    last_num: int = 5  # The image sequence starts with images containing zeros, wait until it is filled

    close_app: bool = False
    model_prediction = np.zeros(1)

    lt: float = 0
    rt: float = 0
    lx: float = 0
    while not close_app:
        try:
            while last_num == img_sequencer.num_sequence:
                time.sleep(0.01)

            last_num = img_sequencer.num_sequence
            img_seq, _ = img_sequencer.get_sequence()

            init_copy_time: float = time.time()

            keys = key_check()
            if "J" not in keys:

                x: torch.tensor = torch.stack(
                    (
                        transform(img_seq[0] / 255.0),
                        transform(img_seq[1] / 255.0),
                        transform(img_seq[2] / 255.0),
                        transform(img_seq[3] / 255.0),
                        transform(img_seq[4] / 255.0),
                    ),
                    dim=0,
                ).to(device=device, dtype=dtype)

                with torch.no_grad():
                    model_prediction: torch.tensor = (
                        model(x, output_mode="keyboard", return_best=True)[0]
                        .cpu()
                        .numpy()
                    )

                select_key(model_prediction)

                key_push_time: float = time.time()

                if show_current_control:
                    var.set("Driving")
                    text_label.config(fg="green")
                    root.update()

                if enable_evasion:
                    score = mse(img_seq[0], img_seq[4])
                    if score < evasion_score:
                        if show_current_control:
                            var.set("Evasion maneuver")
                            text_label.config(fg="blue")
                            root.update()
                        
                        select_key(4)
                        time.sleep(1)
                        if np.random.rand() > 0.5:
                            select_key(6)
                        else:
                            select_key(8)
                        time.sleep(0.2)

                        if show_current_control:
                            var.set("T.E.D.D. 1104 Driving")
                            text_label.config(fg="green")
                            root.update()

            else:
                if show_current_control:
                    var.set("Manual Control")
                    text_label.config(fg="red")
                    root.update()


                key_push_time: float = 0.0

            if show_what_ai_sees:

                # if enable_segmentation:
                #     img_seq = image_segformer.add_segmentation(images=img_seq)

                cv2.imshow("window1", img_seq[0])
                cv2.waitKey(1)
                cv2.imshow("window2", img_seq[1])
                cv2.waitKey(1)
                cv2.imshow("window3", img_seq[2])
                cv2.waitKey(1)
                cv2.imshow("window4", img_seq[3])
                cv2.waitKey(1)
                cv2.imshow("window5", img_seq[4])
                cv2.waitKey(1)

            if "L" in keys:
                time.sleep(0.1)  # Wait for key release
                if show_what_ai_sees:
                    cv2.destroyAllWindows()
                    show_what_ai_sees = False
                else:
                    show_what_ai_sees = True

            time_it: float = time.time() - last_time

            

            info_message = f"Predicted Key: {id_to_key(model_prediction)}"

            print(
                f"Recording at {img_sequencer.screen_recorder.fps} FPS\n"
                f"Actions per second {None if time_it == 0 else 1 / time_it}\n"
                f"Reaction time: {round(key_push_time - init_copy_time, 3) if key_push_time > 0 else 0} secs\n"
                f"{info_message}\n"
                f"Difference from img 1 to img 5 {None if not enable_evasion else score}\n"
                f"Push Ctrl + C to exit\n"
                f"Push L to see the input images\n"
                f"Push J to use to use manual control\n",
                end="\r",
            )

            last_time = time.time()

        except KeyboardInterrupt:
            print()
            img_sequencer.stop()
            close_app = True
if __name__ == "__main__":
    width = 1920
    height = 1080
    full_screen = True
    show_current_control= True
    main(width, height, full_screen, show_current_control)