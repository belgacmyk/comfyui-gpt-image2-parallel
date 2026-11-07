import asyncio
import base64
import io
import json
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from test_gpt_image25 import nodes, schemas, response, BASE

MODEL='gpt-image-2.6-flare'

def op(**kwargs):
    defaults=dict(prompt='test', images=[], mask=None, quality='high', background='opaque',
        n=1, size='auto', moderation='auto', api_base=BASE, auth_token='fake', model=MODEL)
    defaults.update(kwargs)
    return nodes._build_operation(**defaults)

def file_response(content):
    return schemas.OpenAIImageGenerationResponse.model_validate({'data':[{'b64_json':base64.b64encode(content).decode()}]})

class Image26Tests(unittest.TestCase):
    def test_both_models_qualities_and_complexities(self):
        for model in ('gpt-image-2.6-flare','gpt-image-2.6-sunburst'):
            for quality in nodes.GPT_IMAGE_QUALITIES:
                for complexity in ('low','medium','high'):
                    request=op(model=model,quality=quality,complexity=complexity).request.model_dump(mode='json',exclude_none=True)
                    self.assertEqual(request['complexity'],complexity)
                    self.assertEqual(request['quality'],quality)
                    self.assertEqual(request['model'],model)

    def test_auto_omits_new_fields_for_saved_workflows(self):
        for model in (MODEL,'gpt-image-2.5-flare','gpt-image-1.5'):
            data=op(model=model).request.model_dump(mode='json',exclude_none=True)
            for key in ('complexity','frame_count','frame_duration_ms','output_compression'):
                self.assertNotIn(key,data)

    def test_markup_is_rgba_multipart_and_not_mask(self):
        overlay=torch.zeros(1,4,4,4);overlay[:,1,1]=1
        request=op(images=[torch.zeros(1,4,4,3)],markup=overlay,complexity='low')
        self.assertEqual(request.endpoint.path,'images/edits')
        self.assertEqual([key for key,_ in request.files],['image','markup'])
        with Image.open(request.files[1][1]) as im:
            self.assertEqual(im.mode,'RGBA')
            self.assertEqual(im.getchannel('A').getextrema(),(0,255))
        self.assertEqual(request.request.complexity,'low')

    def test_markup_validation_before_network(self):
        image=torch.zeros(1,4,4,3)
        invalid=[dict(markup=torch.zeros(1,4,4,4)),
                 dict(images=[image],markup=torch.ones(1,4,4,4)),
                 dict(images=[image],markup=torch.zeros(1,5,4,4)),
                 dict(images=[image],markup=torch.zeros(1,4,4,3)),
                 dict(images=[image],markup=torch.zeros(1,4,4,4),mask=torch.zeros(1,4,4))]
        for kwargs in invalid:
            with self.subTest(kwargs=list(kwargs)),self.assertRaises(ValueError): op(**kwargs)

    def test_svg_and_gif_contracts(self):
        self.assertEqual(op(output_format='svg').request.output_format.value,'svg')
        gif=op(output_format='gif',images=[torch.zeros(1,4,4,3)],frame_count=12,frame_duration_ms=210)
        data=gif.request.model_dump(mode='json',exclude_none=True)
        self.assertEqual(gif.endpoint.path,'images/edits')
        self.assertEqual((data['frame_count'],data['frame_duration_ms']),(12,210))
        self.assertNotIn('output_compression',data)
        for kwargs in (dict(output_format='svg',n=2),dict(output_format='svg',images=[torch.zeros(1,4,4,3)]),
                       dict(output_format='gif',mask=torch.zeros(1,4,4)),dict(output_format='gif',frame_count=17),
                       dict(output_format='gif',frame_duration_ms=19),dict(output_format='gif',model='gpt-image-2.5-flare'),
                       dict(output_format='jpeg',background='transparent')):
            with self.subTest(kwargs=list(kwargs)),self.assertRaises(ValueError):op(**kwargs)

    def test_compression_only_for_webp_jpeg(self):
        for fmt in ('png','webp','jpeg','svg','gif'):
            data=op(output_format=fmt,output_compression=72).request.model_dump(mode='json',exclude_none=True)
            self.assertEqual(data.get('output_compression'),72 if fmt in ('webp','jpeg') else None)

    def test_raster_parallel_independent_new_controls(self):
        barrier=threading.Barrier(2,timeout=5);seen={}
        def execute(operation):
            seen[operation.request.prompt]=operation.request.model_dump(mode='json',exclude_none=True)
            barrier.wait();return response()
        with patch.object(nodes,'read_user_config',return_value={}),patch.object(nodes.SynchronousOperation,'execute',execute):
            a,b=asyncio.run(nodes.GPTImage2GenerateParallelX2().api_call('A','B',api_base=BASE,complexity_a='low',complexity_b='high',output_format_b='webp'))
        self.assertEqual((seen['A']['complexity'],seen['B']['complexity']),('low','high'))
        self.assertEqual((seen['A']['output_format'],seen['B']['output_format']),('png','webp'))
        self.assertEqual(a.shape[-1],4)

    def test_bad_media_branch_prevents_both_requests(self):
        with patch.object(nodes,'read_user_config',return_value={}),patch.object(nodes.SynchronousOperation,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'n=1'):
                asyncio.run(nodes.GPTImage26FilesParallelX2().api_call('A','B',api_base=BASE,output_format_b='svg',n_b=2))
            execute.assert_not_called()

    def test_media_preserves_svg_and_gif_bytes_and_parallelism(self):
        svg=b'<svg xmlns="http://www.w3.org/2000/svg"><circle r="2"/></svg>'
        buf=io.BytesIO();Image.new('RGB',(4,4),'red').save(buf,format='GIF',save_all=True,append_images=[Image.new('RGB',(4,4),'blue')],duration=200,loop=0)
        gif=buf.getvalue();barrier=threading.Barrier(2,timeout=5)
        def execute(operation):
            barrier.wait();return file_response(svg if operation.request.output_format.value=='svg' else gif)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folders=types.SimpleNamespace(get_output_directory=lambda:tmp,
                get_save_image_path=lambda prefix,out:(tmp,prefix,1,'',prefix))
            with patch.dict('sys.modules',{'folder_paths':folders}),patch.object(nodes,'read_user_config',return_value={}),patch.object(nodes.SynchronousOperation,'execute',execute):
                output=asyncio.run(nodes.GPTImage26FilesParallelX2().api_call('SVG','GIF',api_base=BASE,output_format_a='svg',output_format_b='gif'))
            rows=[json.loads(value)[0] for value in output['result']]
            self.assertEqual((Path(tmp)/rows[0]['filename']).read_bytes(),svg)
            self.assertEqual((Path(tmp)/rows[1]['filename']).read_bytes(),gif)
            with Image.open(Path(tmp)/rows[1]['filename']) as image:self.assertEqual(image.n_frames,2)
            self.assertEqual(len(output['ui']['images']),1)

    def test_single_new_controls_and_no_gif_flattening(self):
        seen=[]
        def execute(operation):seen.append(operation.request);return response()
        with patch.object(nodes,'read_user_config',return_value={}),patch.object(nodes.SynchronousOperation,'execute',execute):
            asyncio.run(nodes.GPTImage1Generate().api_call('test',api_base=BASE,complexity='medium'))
            with self.assertRaisesRegex(ValueError,'Files x2'):
                asyncio.run(nodes.GPTImage1Generate().api_call('test',api_base=BASE,output_format='gif'))
        self.assertEqual(seen[0].complexity,'medium')

if __name__=='__main__': unittest.main()
