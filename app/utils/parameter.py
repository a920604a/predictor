import argparse

def default_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-info-file", type=str, default="./model_info.json",
                        help="input model information json file")
    parser.add_argument("--input-folder-path", type=str, default="./share_data",
                        help="input image folder path")

    parser.add_argument("--result-folder",
                        default="./result", help="result path")
    parser.add_argument('--mode', default='csv', help='csv')
    parser.add_argument("--draw-top1", action="store_true",
                        help="visual image (True of False)")
    parser.add_argument("--draw-total",  action="store_true",
                        help="visual image (True of False)")
    parser.add_argument("--top1", action="store_true",
                        help="whether write top1 info")
    parser.add_argument("--total",  action="store_true",
                        help="whether write total info")
    parser.add_argument(
        "--auto-labelme", action="store_true", help="auto labelme")
    parser.add_argument("--pool-size", type=int, default=4, help="pool size")

    parser.add_argument("--gpus", type=int,
                        default=[0, 1], nargs='+',  help="code")
    args = parser.parse_args()
    return args


# class BoundedThreadPoolExecutor(ThreadPoolExecutor):
#     def __init__(self, max_workers=None, thread_name_prefix=''):
#         super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
#         self._work_queue = queue.Queue(self._max_workers * 2)  # 队列大小为最大线程数的两倍
