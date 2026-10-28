"""Canonical character decoder. Dense matrices use input, output axis order."""
import jax
import jax.numpy as jnp
import numpy as np


def initialize(config, key):
    d, layers = config['n_embd'], config['n_layer']
    shapes = {'token_embedding': (65, d), 'position_embedding': (128, d),
              'final_norm.scale': (d,), 'final_norm.bias': (d,),
              'output.weight': (d, 65), 'output.bias': (65,)}
    for i in range(layers):
        prefix = f'blocks.{i}.'
        for norm in ['norm1', 'norm2']:
            shapes[prefix + norm + '.scale'] = (d,)
            shapes[prefix + norm + '.bias'] = (d,)
        for name, shape in [('attention.qkv', (d, 3*d)), ('attention.output', (d, d)),
                            ('mlp.up', (d, 4*d)), ('mlp.down', (4*d, d))]:
            shapes[prefix + name + '.weight'] = shape
            shapes[prefix + name + '.bias'] = (shape[1],)
    keys = jax.random.split(key, len(shapes))
    return {name: (jnp.ones(shape) if name.endswith('.scale') else
                   jnp.zeros(shape) if name.endswith('.bias') else
                   0.02 * jax.random.normal(k, shape))
            for (name, shape), k in zip(shapes.items(), keys)}


def forward(params, ids, config, *, return_attention=False, causal=True):
    b, t = ids.shape
    d, heads = config['n_embd'], config['n_head']
    # GPU defaults may use TF32; the export contract requires full float32 products.
    def dense(x, name):
        return jnp.matmul(x, params[name + '.weight'], precision=jax.lax.Precision.HIGHEST) + params[name + '.bias']
    def norm(x, name):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean((x - mean)**2, axis=-1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(variance + 1e-5) * params[name + '.scale'] + params[name + '.bias']
    x = params['token_embedding'][ids] + params['position_embedding'][:t]
    first_attention = None
    for i in range(config['n_layer']):
        prefix = f'blocks.{i}.'
        qkv = dense(norm(x, prefix + 'norm1'), prefix + 'attention.qkv')
        q, k, v = [a.reshape(b, t, heads, d//heads).transpose(0, 2, 1, 3)
                   for a in jnp.split(qkv, 3, axis=-1)]
        scores = jnp.matmul(q, k.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST) / np.sqrt(d//heads)
        if causal:
            scores = jnp.where(jnp.arange(t)[:, None] >= jnp.arange(t)[None, :], scores, -jnp.inf)
        attention = jax.nn.softmax(scores, axis=-1)
        if i == 0:
            first_attention = attention
        attended = jnp.matmul(attention, v, precision=jax.lax.Precision.HIGHEST).transpose(0, 2, 1, 3).reshape(b, t, d)
        x = x + dense(attended, prefix + 'attention.output')
        x = x + dense(jax.nn.gelu(dense(norm(x, prefix + 'norm2'), prefix + 'mlp.up'), approximate=False), prefix + 'mlp.down')
    logits = dense(norm(x, 'final_norm'), 'output')
    return (logits, first_attention) if return_attention else logits
