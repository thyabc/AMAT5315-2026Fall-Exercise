"""Run with JAX_PLATFORMS=cpu python -m unittest discover -s week8/speedrun."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from train import optimizer, initialize, save_state, load_state, train, write_log
from model import forward
from eval import metrics
from sample import sample

HERE=Path(__file__).resolve().parent


class TrainingTests(unittest.TestCase):
    def test_checkpoint_and_configuration(self):
        config=json.loads((HERE/'config-tiny.json').read_text())
        key=jax.random.PRNGKey(2)
        params=initialize(config,key)
        tx=optimizer(config)
        state=tx.init(params)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'state.npz'
            save_state(checkpoint,params,state,key,0,1.5,config)
            restored=load_state(checkpoint,tx,config)
            for a,b in zip(jax.tree.leaves((params,state,key)),jax.tree.leaves(restored[:3])):
                np.testing.assert_array_equal(a,b)
            self.assertEqual(restored[3:],(0,1.5))
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                load_state(checkpoint,tx,dict(config,learning_rate=0.1))
            # A checkpoint already at the requested step is a valid no-op resume.
            out=Path(directory)/'out'
            out.mkdir()
            write_log(out/'train_log.csv',[dict(step=1,train_loss=4.,val_loss=4.,tokens=256,wall_seconds=1.,lr=0.)])
            info=train(config,out,Path(directory)/'data',0,checkpoint,evaluate_model=False)
            self.assertIsNone(info['first_update_seconds'])
            self.assertTrue((out/'model.npz').exists())
            self.assertEqual(len((out/'train_log.csv').read_text().splitlines()),1)
            records=sample(out,prompt='To be',seed=7,length=3,temperatures=[0.7,1.1])
            self.assertEqual([r['settings']['temperature'] for r in records],[0.7,1.1])
            for record in records:
                self.assertEqual(len(record['text']),8)
                self.assertTrue(record['text'].startswith('To be'))
                self.assertEqual(record['settings']['length'],3)
                self.assertEqual(record['settings']['total_length'],8)
                self.assertEqual(record['settings']['seed'],7)
            for invalid in [dict(prompt=''),dict(length=-1),dict(temperatures=[0]),dict(temperatures=[float('nan')])]:
                with self.assertRaises(ValueError):
                    sample(out,**invalid)

    def test_log_replacement_failure_preserves_previous_log(self):
        row=dict(step=1,train_loss=4.,val_loss=4.,tokens=256,wall_seconds=1.,lr=0.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'train_log.csv'
            write_log(path,[row])
            previous=path.read_bytes()
            for operation in ['train.os.replace','train.csv.DictWriter.writerows']:
                with patch(operation,side_effect=OSError('simulated interruption')):
                    with self.assertRaises(OSError):
                        write_log(path,[row,dict(row,step=2)])
                self.assertEqual(path.read_bytes(),previous)
            write_log(path,[row,dict(row,step=2)])
            self.assertEqual(len(path.read_text().splitlines()),3)

    def test_causality_and_ppl(self):
        config=json.loads((HERE/'config-tiny.json').read_text())
        params=initialize(config,jax.random.PRNGKey(2))
        a=jnp.arange(16)[None,:]
        b=a.at[:,8:].set(64)
        first,attention=forward(params,a,config,return_attention=True)
        second=forward(params,b,config)
        np.testing.assert_allclose(first[:,:8],second[:,:8],atol=1e-6)
        self.assertEqual(float(jnp.max(jnp.triu(attention,1))),0.)
        unmasked=forward(params,b,config,causal=False)
        self.assertGreater(float(jnp.max(jnp.abs(first[:,:8]-unmasked[:,:8]))),1e-4)
        np.testing.assert_allclose(jax.nn.softmax(first).sum(-1),1,atol=1e-6)
        self.assertAlmostEqual(metrics(-np.log(.5)-np.log(.25),2)['ppl'],np.sqrt(8))
        self.assertAlmostEqual(metrics(np.log(2)+2*np.log(4),3)['ppl'],32**(1/3))


if __name__=='__main__':
    unittest.main()
