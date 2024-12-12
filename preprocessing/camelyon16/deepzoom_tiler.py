import argparse
import yaml
import glob
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Process, JoinableQueue
from unicodedata import normalize

import numpy as np
from PIL import Image, ImageFilter, ImageStat
from shapely.geometry import Polygon

Image.MAX_IMAGE_PIXELS = None

import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

VIEWER_SLIDE_NAME = 'slide'


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                 quality, threshold, **kwargs):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None
        self.tile_label_csv = tile_label_csv
        self.slides_dir = slides_dir
        
        self.tile_label_csv = kwargs['tile_label_csv']
        self.slides_dir = kwargs['slides_dir']
     
        
    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        u = 0
        while True:
            data = self._queue.get()

            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data

            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated

            tile = dz.get_tile(level, address)
            edge = tile.filter(ImageFilter.FIND_EDGES)
            edge = ImageStat.Stat(edge).sum
            edge = np.mean(edge) / (self._tile_size ** 2)
            w, h = tile.size
            if edge > self._threshold:

                (x, y), mlevel = dz.get_tile_coordinates(level, address)[0:2]
                factor = 2 ** mlevel
                
                annotation_file = os.path.join(
                    self.slides_dir, 
                    '/annotations/' + c_slide.split('/')[-1].split('.')[0] + '.xml'
                )
                label = 0
                
                if os.path.isfile(annotation_file):
                    label = self.does_square_have_cancer(
                        annotation_file, x, y, self._tile_size * factor)
                    
                tile_label_csv = open(self.tile_label_csv, "a")
                
                tile_label_csv.write(f'{c_slide}/{outfile},{label}\n')

                if not (w == self._tile_size and h == self._tile_size):
                    tile = tile.resize((self._tile_size, self._tile_size))
                tile.save(outfile, quality=self._quality)

            self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                                 limit_bounds=self._limit_bounds)

    # https://thedeeplearning-intern322.medium.com/extracting-patches-from-whole-slide-images-wsi-c3bb9b00d9b5
    def parse_xml(self, anno_path):
        tree = ET.ElementTree(file=anno_path)
        annolist = {}
        root = tree.getroot()
        i = 0
        for coords in root.iter('Coordinates'):
            vasc = []
            for coord in coords:
                vasc.append((int(float(coord.attrib.get("X"))), int(float(coord.attrib.get("Y")))))
            annolist[i] = vasc
            i += 1
        return annolist

    def does_square_have_cancer(self, annotation_file, x, y, size):
        coordinates = self.parse_xml(annotation_file)
        for i in coordinates:
            polyg1 = Polygon(coordinates[i])
            polyg2 = Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])
            if polyg1.intersects(polyg2):
                return 1

        return 0


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, target_levels, mag_base, format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._target_levels = target_levels
        self._mag_base = int(mag_base)

    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        target_levels = [self._dz.level_count - i - 1 for i in self._target_levels]
        mag_list = [int(self._mag_base / 2 ** i) for i in self._target_levels]
        mag_idx = 0
        for level in range(self._dz.level_count):
            if not (level in target_levels):
                continue
            tiledir = os.path.join("%s_files" % self._basename, str(mag_list[mag_idx]))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            cols, rows = self._dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    tilename = os.path.join(tiledir, '%d_%d-%d.%s' % (
                        col, row, level, self._format))  # added level
                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row),
                                         tilename))
                    self._tile_done()
            mag_idx += 1

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                self._associated or 'slide', count, total),
                  end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""
    def __init__(self, slidepath=None, basename=None, mag_levels=None, base_mag=None, objective=None, 
                 format=None, tile_size=None, overlap=None, limit_bounds=None, quality=None, 
                 workers=None, threshold=None, tile_label_csv=None, **kwargs): 
    # def __init__(self, slidepath, basename, mag_levels, base_mag, objective, format, tile_size, overlap,
    #              limit_bounds, quality, workers, threshold, tile_label_csv):
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mag = base_mag
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._dzi_data = {}
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                       limit_bounds, quality, threshold, **kwargs).start()

    def run(self):
        self._run_image()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                               limit_bounds=self._limit_bounds)

        MAG_BASE = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if MAG_BASE is None:
            MAG_BASE = self._objective
        first_level = int(math.log2(float(MAG_BASE) / self._base_mag))  # raw / input, 40/20=2, 40/40=0
        target_levels = [i + first_level for i in self._mag_levels]  # levels start from 0
        target_levels.reverse()

        tiler = DeepZoomImageTiler(dz, basename, target_levels, MAG_BASE, self._format, associated,
                                   self._queue)
        tiler.run()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()


def nested_patches(img_slide=None, out_base=None, level=(0,), wsi_temp_folder=None, ext='jpeg'):
    print('\n Organizing patches')
    img_name = img_slide.split(os.sep)[-1].split('.')[0]
    img_class = img_slide.split(os.sep)[2]
    n_levels = len(glob.glob(f'{wsi_temp_folder}/*'))
    bag_path = os.path.join(out_base, img_class, img_name)
    os.makedirs(bag_path, exist_ok=True)
    
    if len(level) == 1:
        patches = glob.glob(os.path.join(wsi_temp_folder, '*', '*.' + ext))
        for i, patch in enumerate(patches):
            patch_name = patch.split(os.sep)[-1]
            shutil.move(patch, os.path.join(bag_path, patch_name))
            sys.stdout.write('\r Patch [%d/%d]' % (i + 1, len(patches)))
        print('Done.')
        
    else:
        level_factor = 2 ** int(level[1] - level[0])
        levels = [int(os.path.basename(i)) for i in glob.glob(os.path.join(wsi_temp_folder, '*'))]
        levels.sort()
        low_patches = glob.glob(os.path.join(wsi_temp_folder, str(levels[0]), '*.' + ext))
        
        for i, low_patch in enumerate(low_patches):
            low_patch_name = low_patch.split(os.sep)[-1]
            shutil.move(low_patch, os.path.join(bag_path, low_patch_name))
            low_patch_folder = low_patch_name.split('.')[0]
            high_patch_path = os.path.join(bag_path, low_patch_folder)
            os.makedirs(high_patch_path, exist_ok=True)
            low_x = int(low_patch_folder.split('_')[0])
            low_y = int(low_patch_folder.split('_')[1])
            high_x_list = list(range(low_x * level_factor, (low_x + 1) * level_factor))
            high_y_list = list(range(low_y * level_factor, (low_y + 1) * level_factor))
            
            for x_pos in high_x_list:
                for y_pos in high_y_list:
                    high_patch = glob.glob(
                        os.path.join(wsi_temp_folder, str(levels[1]), '{}_{}.'.format(x_pos, y_pos) + ext))
                    if len(high_patch) != 0:
                        high_patch = high_patch[0]
                        shutil.move(high_patch, os.path.join(bag_path, low_patch_folder, high_patch.split(os.sep)[-1]))
            os.rmdir(os.path.join(bag_path, low_patch_folder))
            os.remove(low_patch)
            sys.stdout.write('\r Patch [%d/%d]' % (i + 1, len(low_patches)))
            
        print('Done.')

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='Patch extraction for camelyon16')
    parser.add_argument(
        '-c', '--config', type=str, default = './configs/deepzoom_tiler.yaml', help='Path to the config.yaml file'
    ) 
    parser.add_argument(
        '-d', '--dataset', type=str,
        default='camelyon16',  # Changed to camelyon16
        help='Dataset name'
    )
    parser.add_argument('-e', '--overlap', type=int, default=0, help='Overlap of adjacent tiles [0]')
    parser.add_argument('-f', '--format', type=str, default='jpeg', help='Image format for tiles [jpeg]')
    parser.add_argument(
        '-v', '--slide_format', type=str, default='tif',
        help='Image format for tiles [svs]'  # changed to TIF (for Camelyon16)
    )
    parser.add_argument('-j', '--workers', type=int, default=8,
                        help='Number of worker processes to start [4]')  # 12 because i have many cores
    parser.add_argument(
        '-q', '--quality', type=int, default=75,
        help='JPEG compression quality [70]'  # We changed the default quality to 75, the default of pillow
    )
    parser.add_argument(
        '-s', '--tile_size', type=int, default=256,
        help='Tile size [256]'  # We changed the default to 256 for a more accurate comparison with other methods
    )
    parser.add_argument(
        '-b', '--base_mag', type=float, default=20,
        help='Maximum magnification for patch extraction [20]'
    )
    parser.add_argument(
        '-o', '--objective', type=float, default=20,
        help='The default objective power if metadata does not present [20]'
    )
    parser.add_argument(
        '-t', '--background_t', type=int, default=20,
        help='Threshold for filtering background [15]'  # Changed to 20
    )
    parser.add_argument("--tile_label_csv", type=str)
    parser.add_argument("--wsi_temp_folder", type=str)
    
    args = parser.parse_args()

    config = load_config(args.config)
    
    args.slide_format = args.slide_format or config['SLIDE_FORMAT']
    args.workers = args.workers or config['WORKERS']
    args.quality = args.quality or config['QUALITY']
    args.tile_size = args.tile_size or config['TILE_SIZE']
    args.base_mag = args.base_mag or config['BASE_MAG']
    args.objective = args.objective or config['OBJECTIVE']
    args.background_t = args.background_t or config['BACKGROUND_T']
    args.dataset = args.dataset or config['DATASET']
    args.format = args.format or config['FORMAT']
    args.slides_dir = args.slides_dir or config['SLIDES_DIR']
    args.output_dir = args.output_dir or config['OUTPUT_DIR']
    args.tile_label_csv = args.tile_label_csv or config['TILE_LABEL_CSV'] 
    args.wsi_temp_folder = args.args_temp_folder or config['WSI_TEMP_FOLDER']
    
    levels = [config['LEVELS']]
    print("image_dir: ", args.slides_dir)
    out_base = os.path.join(args.output_dir, 'single')
    all_slides = glob.glob(os.path.join(args.slides_dir, '0_normal/*.' + args.slide_format)) + \
                 glob.glob(os.path.join(args.slides_dir, '1_tumor/*.' + args.slide_format))

    # pos-i_pos-j -> x, y
    print(tile_label_csv)
    tile_label_csv = open(args.tile_label_csv, "a")
    tile_label_csv.write('slide_name,label\n')
    tile_label_csv.close()
    
    temp_dir = args.wsi_temp_folder
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Directory {temp_dir} already existed and has been removed.")
 
    for idx, c_slide in enumerate(all_slides):
        print('Process slide {}/{} : {}'.format(idx + 1, len(all_slides), c_slide))
        tiler = DeepZoomStaticTiler(
            slidepath=c_slide,
            basename="WSI_temp",
            mag_levels=levels,
            base_mag=args.base_mag,
            objective=arg.objective,
            format=format ,
            tile_size=tile_size, #256,
            overlap=overlap,
            limit_bounds=True,
            quality=args.quality, #75,
            workers=args.workers, #8,
            threshold=background_t,#0.5,
            tile_label_csv=args.tile_label_csv,
            slides_dir=args.slides_dir, 
            **kwargs
        ).run()
        
        nested_patches(
            img_slide=c_slide, 
            out_base=out_base, 
            levels=levels, 
            wsi_temp_folder=wsi_temp_folder, 
            ext=args.format)
        
        shutil.rmtree(temp_dir)
        
    tile_label_csv.close()
    print('Patch extraction done for {} slides.'.format(len(all_slides)))