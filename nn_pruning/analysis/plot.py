from matplotlib import pyplot as pyplot
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png
import json
from pathlib import Path
import re
import shutil
import plot_data
import math

import bokeh
import bokeh.plotting as plotting
from bokeh.models import Label, Range1d
from bokeh.resources import CDN
from bokeh.embed import autoload_static
import itertools


class Plotter:
    def __init__(self, dest_dir, dest_file_name, only_dots, draw_labels, limits, title, x_label, y_label):
        self.dest_dir = dest_dir
        self.dest_file_name = dest_file_name
        self.only_dots = only_dots
        self.draw_labels = draw_labels
        self.limits = limits
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    @staticmethod
    def label_cleanup(label):
        for s in " _,/=":
            label = label.replace(s, "_")
        return label

class MatplotlibPlotter(Plotter):
    def save_fig(self, extension):
        pyplot.savefig(
            self.dest_dir / (self.dest_file_name + "." + extension),
            dpi=None,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=None,
            transparent=False,
            bbox_inches=None,
            pad_inches=0.1,
            metadata=None,
        )

    def run(self, plots, key):
        self.fontsize = 15
        draw_labels = self.draw_labels

        x_min = self.limits[key]["x_min"]
        x_max = self.limits[key]["x_max"]
        y_min = self.limits[key]["y_min"]
        y_max = self.limits[key]["y_max"]

        markers = ["o", "v", "s", "+", "v"]

        fig = pyplot.figure(figsize=(20, 12))
        ax1 = fig.add_subplot(111)
        max_x = 0

        for label_text, plot in plots.items():
            x = plot["x"]
            y = plot["y"]
            annotate = plot["annotate"]
            max_x = max(max_x, max(x))

            if len(x) == 1 or self.only_dots:
                pyplot.scatter(x, y, cmap="viridis", alpha=1.0, label=label_text)  # , marker=markers[i]) # cool
            else:
                pyplot.plot(x, y, label=label_text)  # , marker=markers[i]) # cool

            if draw_labels or len(x) == 1:
                for i, txt in enumerate(x):
                    if x_min is None or x[i] >= x_min and x[i] <= x_max:
                        if y_min is None or y[i] >= y_min and y[i] <= y_max:
                            annotate = annotate or label_text
                            ax1.annotate(annotate, (x[i] + 0.005, y[i] + 0.005))

        for i in range(3):
            f1 = 88.5 - i
            pyplot.plot(
                [0, 10], [88.5 - i] * 2, label=f"original BERT-base {'' if i == 0 else str(-i) + '% '}(F1 = {f1})"
            )

        legend_pos = self.limits[key]["legend_pos"]
        pyplot.legend(loc=legend_pos, prop={"size": self.fontsize})

        if x_min != None:
            pyplot.xlim(x_min, x_max)
        if y_min != None:
            pyplot.ylim(y_min, y_max)

        pyplot.xlabel(self.x_label, fontsize=self.fontsize)
        pyplot.ylabel(self.y_label, fontsize=self.fontsize)
        pyplot.title(self.title, fontsize=self.fontsize)

        self.save_fig("png")
        self.save_fig("eps")


class BokehHelper:
    def __init__(self, div_id, js_path, only_dots):
        self.div_id = div_id
        self.js_path = js_path
        self.only_dots = only_dots

    def create_fig(self):
        raise RuntimeError("Please implement in subclass")

    def run(self, *args, show=False, **kwargs):
        # html = file_html(self.fig, CDN, self.div_id)
        if show:
            plotting.output_notebook()
        fig = self.create_fig(*args, **kwargs)

        if show:
            bokeh.plotting.show(fig)
        else:
            js, tag = autoload_static(fig, CDN, self.js_path)
            return fig, js, tag

class BokehPlot(BokehHelper):
    def get_legend_pos(self, limits):
        replacements = [(" ", "_"), ("upper", "top"), ("lower", "bottom")]
        pos = limits["legend_pos"]
        for s,d in replacements:
            pos = pos.replace(s,d)
        return pos

    def create_fig(self, plots, key, title, width, x_label, y_label, limits):
        # select a palette
        from bokeh.palettes import Dark2_5 as palette

        # create a new plot with a title and axis labels
        fig = bokeh.plotting.figure(title=title, width = width, x_axis_label='x', y_axis_label='y')
        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label

        limits = limits[key]
        x_min = limits["x_min"]
        x_max = limits["x_max"]
        y_min = limits["y_min"]
        y_max = limits["y_max"]

        fig.x_range = Range1d(x_min, x_max)
        if y_min is not None and y_max is not None:
            fig.y_range = Range1d(y_min, y_max)

        # add a line renderer with legend and line thickness

        colors = itertools.cycle(palette)

        max_x = -math.inf
        for plot in plots.values():
            max_x = max(max_x, max(plot["x"]))

        for (label_text, plot), color in zip(plots.items(), colors):
            x = plot["x"]
            y = plot["y"]
            annotate = plot["annotate"]

            if len(x) == 1 or self.only_dots:
                point = Label(x=x[0], y=y[0], text = label_text)
                fig.add_layout(point)
                fig.scatter(x, y)  # , marker=markers[i]) # cool
            else:
                fig.line(x, y, legend_label=label_text, line_width=2, color=color)

            base_f1 = 88.5
            for i in range(3):
                f1 = base_f1 - i
                line_args = dict(legend_label=f"Reference f1={base_f1} BERT-base")

                fig.line(
                    x = [0, max_x],
                    y = [88.5 - i] * 2,
                    color="red",
                    alpha=0.5**(i + 2),
                    **line_args
                )

            if False:
                if draw_labels or len(x) == 1:
                    for i, txt in enumerate(x):
                        if x_min is None or x[i] >= x_min and x[i] <= x_max:
                            if y_min is None or y[i] >= y_min and y[i] <= y_max:
                                annotate = annotate or label_text
                                ax1.annotate(annotate, (x[i] + 0.005, y[i] + 0.005))

        fig.legend.click_policy = "hide"

        fig.legend.location = self.get_legend_pos(limits)
        return fig


class BokehPlotter(Plotter):

    def save_file(self, extension, data):
        with (self.dest_dir / (self.dest_file_name + "." + extension)).open("w") as f:
            f.write(data)

    def run(self, plots, key):
        WIDTH=800
        div_id =  str(self.dest_file_name)
        js_path = "$$JS_PATH$$"
        bp = BokehPlot(div_id, js_path, only_dots = self.only_dots)
        fig, js, tag = bp.run(plots,
                              key,
                              width=WIDTH,
                              title = self.title,
                              x_label=self.x_label,
                              y_label=self.y_label,
                              limits = self.limits)

        export_png(fig, filename=self.dest_dir / (self.dest_file_name + "_static.png"), width=WIDTH)

        self.save_file("js", js)
        self.save_file("html", tag)

class TextFilePlotter(Plotter):
    def log_plot(self, dest_file_name, x, y, data, x_key, y_key):
        with dest_file_name.open("w") as f:
            for i in range(len(x)):
                s = {x_key: x[i], y_key: y[i], "meta": data[i]}
                f.write(json.dumps(s) + "\n")

    def run(self, plots, key):

        for label_text, plot in plots.items():
            x = plot["x"]
            y = plot["y"]
            point = plot["points"]
            annotate = plot["annotate"]

            label_for_file = self.label_cleanup(label_text)
            log_dir = self.dest_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            self.log_plot(log_dir / f"{self.dest_file_name}_{label_for_file}.jsonl", x, y, point, key, "f1")


class PlotManager:
    LIMITS = {
        "speedup": dict(legend_pos="upper right", x_min=0.95, x_max=4.0, y_min=84, y_max=90),
        "fill_rate": dict(legend_pos="lower right", x_min=0.0, x_max=0.75, y_min=84, y_max=90),
    }

    def __init__(
        self,
        task,
        input_filename,
        cat_fun_names=None,
        print_misc=True,
        draw_labels=False,
        convex_envelop=True,
        only_dots=False,
        fontsize=15,
        white_list=None,
        black_list=None,
        reference_black_list=None,
        limits=None,
        label_mapping=None,
    ):
        self.cache_dir = Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.task = task

        self.input_filename = input_filename
        self.cat_fun_names = cat_fun_names or []
        self.print_misc = print_misc
        self.draw_labels = draw_labels
        self.convex_envelop = convex_envelop
        self.only_dots = only_dots or not convex_envelop
        self.fontsize = fontsize
        self.white_list = white_list
        self.black_list = black_list

        self.limits = limits or self.LIMITS

        self.reference_black_list = reference_black_list
        self.label_mapping = label_mapping or {}

    def convexity_filter_checkpoints(self, checkpoints, key="speedup", filt=True):
        margin = 1.0001
        if key == "fill_rate":
            sgn = -1
        else:
            sgn = 1
        key_margin_min = 0.001
        sort_by_key = lambda x: sgn * x.get(key, 1.0)
        checkpoints.sort(key=sort_by_key, reverse=True)
        best_c = checkpoints[0]

        filtered_checkpoints = [best_c]
        for checkpoint in checkpoints[1:]:
            f1_margin = checkpoint["f1"] > filtered_checkpoints[-1]["f1"] * margin
            key_margin = abs(checkpoint[key] - filtered_checkpoints[-1][key])
            if key_margin < key_margin_min:
                if checkpoint["f1"] > filtered_checkpoints[-1]["f1"]:
                    filtered_checkpoints[-1] = checkpoint
            else:
                if not filt or f1_margin:
                    filtered_checkpoints.append(checkpoint)

        filtered_checkpoints.sort(key=sort_by_key, reverse=False)

        return filtered_checkpoints

    def match_regexp_list(self, key, regexps):
        for r in regexps:
            if re.fullmatch(r, key):
                return True
        return False


    def plot(self, key, dest_dir, dest_file_name):
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(exist_ok=True)

        references, checkpoints = plot_data.MultiProvider().points(self.task, self.cache_dir, self.input_filename)

        final_plots = {}

        reference_black_list = self.reference_black_list
        for k, v in references.items():
            if reference_black_list is not None and k in reference_black_list:
                continue

            final_plots[k] = v

        for k, v in checkpoints.items():
            if self.white_list is not None and not self.match_regexp_list(k, self.white_list):
                print(f"Rejected (WL) {k}")
                continue
            if self.black_list is not None and self.match_regexp_list(k, self.black_list):
                print(f"Rejected (BL) {k}")
                continue
            v = checkpoints[k]
            final_plots[k] = v

        max_x = -math.inf
        plots = {}
        for label, data in final_plots.items():
            x0 = [e.get(key, 1.0) for e in data]

            if self.convex_envelop:  # and len(set(x0)) > 1:
                data = self.convexity_filter_checkpoints(data, key=key, filt=len(set(x0)) > 1)

            x = [e.get(key, 1.0) for e in data]
            max_x = max(max_x, max(x))
            y = [e["f1"] for e in data]
            annotate = [e.get("annotate","") for e in data]

            label_text = self.label_mapping.get(label, label.capitalize())
            plots[label_text] = {"x":x, "y":y, "annotate": annotate, "points":data}

        x_label = key.capitalize()
        y_label = "F1"

        title = "%s against %s (BERT-base reference)" % (y_label, x_label)

        params = dict(dest_dir=dest_dir,
                      dest_file_name=dest_file_name,
                      only_dots=self.only_dots,
                      draw_labels=self.draw_labels,
                      limits=self.limits,
                      title=title,
                      x_label=x_label,
                      y_label=y_label)

        constructors = [MatplotlibPlotter, TextFilePlotter, BokehPlotter]

        for c in constructors:
            m = c(**params)
            m.run(plots, key)




class GeneralPlotter(PlotManager):
    #    select = ['block_sparse', 'improved soft movement with distillation', 'misc', 'new method, attention pruned with rows', 'new method, longer final warmup (15 instead of 10)', 'new xp, block_size= 16x16', 'new xp, block_size= 32x32', 'new xp, block_size= 64x32', 'small_epoch']

    CAT_FUN_NAMES = [
        "new_xp",
        "full_block",
        "improved_mvmt_pruning",
        "longer_final_warmup",
        "attention_rows",
        "short_final_warmup",
        "block_unstructured",
    ]
    BLACK_LIST = ["misc"]  # "small_epoch", "new method, attention pruned with rows",
    WHITE_LIST = [
        "improved soft movement with distillation",
        "Structured pruning",
        "Block/struct method, bs= 32x32, v=0",
        "Block/struct method, bs= 32x32, v=1",
        "Block/struct method, bs= 16x16, v=1",
        "Full block method, bs= 32x32, v=1",
        "new xp, block_size= 64x768",
    ]
    REFERENCE_BLACK_LIST = ["local_movement_pruning", "soft_movement_with_distillation"]

    def __init__(
        self,
        task,
        input_file_name,
        cat_fun_names=None,
        cluster=False,
        white_list=False,
        black_list=False,
        reference_black_list=None,
        **kwargs
    ):
        cat_fun_names = cat_fun_names or self.CAT_FUN_NAMES
        if white_list is False:
            white_list = None
        else:
            white_list = white_list or self.WHITE_LIST
        if black_list is False:
            black_list = None
        else:
            black_list = black_list or self.BLACK_LIST

        self.cluster = cluster
        if reference_black_list is False:
            reference_black_list = None
        else:
            reference_black_list = reference_black_list or self.REFERENCE_BLACK_LIST

        super().__init__(
            task,
            input_file_name,
            cat_fun_names=cat_fun_names,
            white_list=white_list,
            black_list=black_list,
            reference_black_list=reference_black_list,
            **kwargs,
        )


def multiplot(p, name):
    name = Plotter.label_cleanup(name)
    dest = Path("graphs") / name
    p.plot("speedup", dest, f"{name}_speedup")
    p.plot("fill_rate", dest, f"{name}_fill_rate")


def draw_all_plots(input_file_name, task, cleanup_cache=False):
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    if cleanup_cache:
        xp_file = cache_dir / "experiments_squadv1.json"
        if xp_file.exists():
            xp_file.unlink()

    plots = {
        "summary": dict(
            cat_fun_names=["new_xp"],
            draw_labels=False,
            #reference_black_list=["local_movement_pruning"],
            white_list=["Block/struct method, final fine tuned, s=[bl]", "Structured pruning",  "improved soft movement with distillation"],
            label_mapping={
                "Block/struct method, final fine tuned, s=l": "Hybrid pruning, BERT-large",
                "Block/struct method, final fine tuned, s=b": "Hybrid pruning, BERT-base",
                "Structured pruning": "Structured pruning, BERT-base",
                "improved soft movement with distillation": "Improved soft movement, BERT-base",
                "soft_movement_with_distillation" : "Original Soft Movement",
            },
        ),
        }
    for name, configuration in plots.items():
        limits = {
            "speedup": dict(legend_pos="upper right", x_min=0.75, x_max=4.0, y_min=None, y_max=None),
            "fill_rate": dict(legend_pos="upper left", x_min=0.0, x_max=0.9, y_min=None, y_max=None),
        }
        p = GeneralPlotter(
            task,
            input_file_name,
            limits=limits,
            cluster=True,
            **configuration,
        )
        multiplot(p, name)

def copy_plots(task):
    graph_path = (Path(__file__).parent / "graphs").resolve()
    dest_dir = (Path(__file__).parent.parent.parent / "docs" / "assets" / "media" / task).resolve()

    parts = ["summary"]
    suffixes = [".html", ".js", ".png"]
    for part in parts:
        part_dir = graph_path / part
        for name in part_dir.iterdir():
            if name.suffix in suffixes:
                src = name
                dst = str(dest_dir / name.name)
                shutil.copyfile(src, dst)

#

if __name__ == "__main__":
    input_file_name = "test.json"
    task = "squadv1"

    draw_all_plots(input_file_name, task, cleanup_cache=False)
    copy_plots(task)
    import sys

    sys.exit()

    white_list = [  # "Block/struct method, bs= [0-9]+x[0-9]+, v=1",
        "Block/struct method, bs= 32x32, v=1",
        "Block/struct method, final fine tuned, s=[bl]",
        "Block/unstruct method, bs= [0-9]+x[0-9]+",
        "improved soft movement with distillation",
        "soft_movement_with_distillation",
        "Full block method, bs= [0-9]+x[0-9]+",
        "Structured pruning",
    ]
    black_list = ["new method, attention pruned with rows"]  # "Block/struct method, bs= [0-9]+x.[0-9]+, v=0"]
    raw_black_list = copy.deepcopy(white_list)
    raw_black_list += copy.deepcopy(black_list)
    # raw_black_list = False

    limits = {
        "speedup": dict(legend_pos="upper right", x_min=0.75, x_max=4.0, y_min=None, y_max=None),
        "fill_rate": dict(legend_pos="upper left", x_min=0.0, x_max=0.75, y_min=None, y_max=None),
    }

    # For debug purpose : black_list is the kept white list
    p = GeneralPlotter(
        task,
        input_file_name,
        white_list=False,
        black_list=raw_black_list,
        reference_black_list=None,
        limits=limits,
    )
    multiplot(p, "raw_global")
    #    sys.exit(0)
    reference_black_list = ["local_movement_pruning"]
    p = GeneralPlotter(
        task,
        input_file_name,
        white_list=white_list,
        black_list=black_list,
        reference_black_list=reference_black_list,
        limits=limits,
    )
    multiplot(p, "global")

    CAT_FUN_NAMES = {
        #    "new_xp_v0": dict(fun_name="new_xp", draw_labels=False, white_list=["Block/struct method, bs= [0-9]+x.[0-9]+, v=0"]),
        "new_xp_v0": dict(
            cat_fun_names=["new_xp"],
            draw_labels=False,
            white_list=["Block/struct method, final fine tuned, s=[bl]", "Structured pruning"],
            label_mapping=new_xp_v0_mapping,
        ),
        "new_xp_v1": dict(
            cat_fun_names=["new_xp"],
            draw_labels=True,
            white_list=[
                "Block/struct method, bs= 32x32, v=1, s=[bl]",
                "Block/struct method, final fine tuned, s=[bl]",
            ],
        ),
        "structured": dict(cat_fun_names=["new_xp"], draw_labels=False, white_list=["Structured pruning"]),
        #        "new_xp_16": dict(fun_name="new_xp", draw_labels=False, white_list=["Block/struct method, bs= 16x16, v=[0-9]+"]),
        #        "new_xp_32": dict(fun_name="new_xp", draw_labels=False, white_list=["Block/struct method, bs= 32x32, v=[0-9]+"]),
        "full_block": {},
        "improved_mvmt_pruning": {},
        "block_unstructured": {},
    }
    for name, configuration in CAT_FUN_NAMES.items():
        limits = {
            "speedup": dict(legend_pos="upper right", x_min=0.75, x_max=4.0, y_min=None, y_max=None),
            "fill_rate": dict(legend_pos="upper left", x_min=0.0, x_max=0.75, y_min=None, y_max=None),
        }
        p = GeneralPlotter(
            task,
            input_file_name,
            limits=limits,
            cluster=True,
            **configuration,
        )
        multiplot(p, name)
#        break