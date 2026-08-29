// WebGL and CA logic
const canvas = document.getElementById('c');
const glsl = SwissGL(canvas);

let lastDrawTime = 0;
let CAs;
let CA1 = null;
let CA2 = null;
let CA_state;
let frame_count = 0;

// The squeezing rules of arxiv:2509.20730. Each name has an "<name> OR" and an
// "<name> AND" 512-bit code in ca_rules.json.
const RULES = ["R2", "R3", "F", "M", "T"];

const params = {
    rule: "R2",
    grid_size: 200,
    run_ca: true,
    steps_per_frame: 2,
};

const uniforms = {
    init_bit: 0,
    brush_bit: 1,
    noise_prob: 0.01,
    noise_bias: 0.0,
    update_prob: 0.25,
    rule_prob: 0.5,
    brush_size: 0.15,
    mouse_x: 0.0,
    mouse_y: 0.0,
    mouse_down: false,
};

// UI Event Handlers
function setupUIControls() {
    // Grid size slider
    const gridSizeSlider = document.getElementById('grid-size');
    const gridSizeValue = document.getElementById('grid-size-value');
    gridSizeSlider.addEventListener('input', (e) => {
        params.grid_size = parseInt(e.target.value);
        gridSizeValue.textContent = e.target.value;
        reset_state();
    });

    // Run CA toggle
    play_pause_event = () => {
        document.getElementById("play").style.display = params.run_ca ? "inline" : "none";
        document.getElementById("pause").style.display = !params.run_ca ? "inline" : "none";
        params.run_ca = !params.run_ca;
    }
    document.getElementById('play').addEventListener("click", play_pause_event);
    document.getElementById('pause').addEventListener("click", play_pause_event);


    const initBitSelect = document.getElementById("init-bit-select");
    initBitSelect.querySelectorAll("input").forEach((sel, i) => {
        sel.onchange = () => {
            uniforms.init_bit = (i != 2) ? i : -1; // -1 for random initialization
            reset_state();
        }
    });

    const brushBitSelect = document.getElementById("brush-bit-select");
    brushBitSelect.querySelectorAll("input").forEach((sel, i) => {
        sel.onchange = () => {
            uniforms.brush_bit = (i != 2) ? i : -1; // -1 for random brush
        }
    });

    const brushSizeSelect = document.getElementById("brush-size-select");
    brushSizeSelect.querySelectorAll("input").forEach((sel, i) => {
        sel.onchange = () => {
            uniforms.brush_size = [0.05, 0.15, 0.3][i];
        }
    });


    // Sliders
    const sliders = [
        {id: 'noise-prob', param: 'noise_prob', uniform: true},
        {id: 'noise-bias', param: 'noise_bias', uniform: true},
        {id: 'update-prob', param: 'update_prob', uniform: true},
        {id: 'rule-prob', param: 'rule_prob', uniform: true},
        {id: 'spf', param: 'steps_per_frame', uniform: false}
    ];

    const text_inputs = [
        {id: 'noise-prob-input', param: 'noise_prob', uniform: true},
        {id: 'noise-bias-input', param: 'noise_bias', uniform: true},
        {id: 'rule-prob-input', param: 'rule_prob', uniform: true},
    ]

    sliders.forEach(({id, param, uniform}) => {
        const slider = document.getElementById(id);
        const valueDisplay = document.getElementById(id + '-value');

        slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (uniform) {
                uniforms[param] = value;
            } else {
                params[param] = value;
            }

            if (id === 'spf') {
                valueDisplay.textContent = ['1/60x', '1/30x', '1/10x', '1/3x', '1x', '2x', '4x', '8x', '16x'][parseInt(e.target.value) + 4];
            } else {
                valueDisplay.textContent = value;
            }
        });
    });

    text_inputs.forEach(({id, param, uniform}) => {
        const input = document.getElementById(id);
        input.addEventListener('change', (e) => {
            let value = parseFloat(e.target.value);
            if (isNaN(value)) {
                value = 0.0;
            }
            e.target.value = value;
            if (uniform) {
                uniforms[param] = value;
            } else {
                params[param] = value;
            }
            const slider = document.getElementById(id.replace('-input', ''));
            slider.value = value;
            const valueDisplay = document.getElementById(id.replace('-input', '') + '-value');
            valueDisplay.textContent = value;
        });
    });

    // Rule selection: picking a rule loads its OR/AND pair together
    const ruleSelect = document.getElementById('rule-select');
    ruleSelect.addEventListener('change', (e) => {
        params.rule = e.target.value;
        load_rule_pair();
    });
}

// Rule 1 is the OR half of the selected rule, rule 2 the AND half.
function load_rule_pair() {
    CA1 = load_CA(CAs[`${params.rule} OR`].code, "rule1");
    CA2 = load_CA(CAs[`${params.rule} AND`].code, "rule2");
}

async function init() {
    const response = await fetch("ca_rules.json");
    CAs = await response.json();

    // Populate rule select
    const ruleSelect = document.getElementById('rule-select');
    ruleSelect.innerHTML = '';
    RULES.forEach(label => {
        const option = document.createElement('option');
        option.value = label;
        option.textContent = label;
        ruleSelect.appendChild(option);
    });
    ruleSelect.value = params.rule;

    load_rule_pair();
    reset_state();
    setupUIControls();
    render();
}

function load_CA(code, tag = "rule") {
    binary_code = Float32Array.from(code, (c) => c === '1' ? 1.0 : 0.0)
    let ca = {
        code: code,
        binary_code: binary_code,
        rule_bits: glsl({}, {
            size: [1, 512],
            format: "r32f",
            story: 1,
            tag: tag,
            data: binary_code
        }),
    }
    return ca;
}

function brush() {
    glsl({
        ...uniforms,
        seed: Math.random() * 5132,
        FP: `
                    float d = distance(UV, vec2(mouse_x, mouse_y));
                    if (d < brush_size) {
                        if (brush_bit == -1.0) {
                            float b = hash(ivec3(I, seed)).x;
                            FOut = vec4(b < 0.5 ? 1.0: 0.0);
                        } else if (brush_bit == 1.0) {
                            FOut = vec4(1);
                        } else {
                            FOut = vec4(0);
                        }
                    } else {
                        FOut = Src(I);
                    }
                `
    }, CA_state);
}

function reset_state() {
    CA_state = glsl({
        seed: Math.random() * 1000, ...uniforms,
        FP: `
                    if (init_bit == -1.0) {
                        float b = hash(ivec3(I, seed)).x;
                        FOut = vec4(b < 0.5 ? 1.0: 0.0);
                    } else if (init_bit == 1.0) {
                        FOut = vec4(1);
                    } else {
                        FOut = vec4(0);
                    }
                `
    }, {size: [params.grid_size, params.grid_size], format: 'r16f', story: 2, tag: 'state'});
}

// Reset Button
document.getElementById('reset_state').addEventListener('click', () => {
    reset_state();
});

// Mouse click
canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    if (e.button === 0) {
        uniforms.mouse_down = true;
        uniforms.mouse_x = e.offsetX / canvas.width;
        uniforms.mouse_y = 1.0 - e.offsetY / canvas.height;
        brush();
    }
});
canvas.addEventListener('mouseup', (e) => {
    e.preventDefault();
    if (e.button === 0) {
        uniforms.mouse_down = false;
    }
});
canvas.addEventListener('mousemove', (e) => {
    e.preventDefault();
    uniforms.mouse_x = e.offsetX / canvas.width;
    uniforms.mouse_y = 1.0 - e.offsetY / canvas.height;
    if (uniforms.mouse_down) {
        brush();
    }
});

// Touch events
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    uniforms.mouse_down = true;
});
canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    uniforms.mouse_down = false;
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    uniforms.mouse_x = (touch.clientX - rect.left) / canvas.width;
    uniforms.mouse_y = 1.0 - (touch.clientY - rect.top) / canvas.height;
    if (uniforms.mouse_down) {
        brush();
    }
});

function step(t) {
    if (!params.run_ca) return;

    glsl({
        ...uniforms,
        rule1: CA1.rule_bits[0],
        rule2: CA2.rule_bits[0],
        seed: t + Math.random() * 6523,
        FP: `
                    float s = Src(I).x;
                    float p = 1.0;
                    float res = 0.0;
                    bool update_flag = hash(ivec3(I, seed)).x < update_prob;
                    bool rule_flag = hash(ivec3(I, seed + 5311.0)).x < rule_prob;

                    if (!update_flag) {
                        FOut = vec4(s);
                        return;
                    }

                    bool noise_flag = hash(ivec3(I, seed + 1231.0)).x < noise_prob;
                    if (noise_flag) {
                        bool bias_flag = (hash(ivec3(I, seed + 7861.0)).x - 0.5) * 2.0 < noise_bias;
                        if (bias_flag) {
                            FOut = vec4(1.0);
                        } else  {
                            FOut = vec4(0.0);
                        }
                    } else {
                        for (int i = -1; i < 2; i++) {
                            for (int j = -1; j < 2; j++) {
                                ivec2 pos = (I + ivec2(i,j)+ViewSize)%ViewSize;
                                res += Src(pos).x * p;
                                p *= 2.0;
                            }
                        }
                        float s_next = rule_flag ? rule1(ivec2(0, int(res))).x : rule2(ivec2(0, int(res))).x;
                        FOut = vec4(s_next);
                    }
                `
    }, CA_state);
}

function render(t) {
    if (!CA1 || !CA2) return;

    frame_count++;
    let spf = params.steps_per_frame;
    let steps = 1;
    if (spf <= 0) {
        const skip = [1, 3, 10, 30, 60][-spf]
        steps = (frame_count % skip) ? 0 : 1;
    } else {
        steps = [1, 2, 4, 8, 16][spf]
    }

    for (let i = 0; i < steps; i++) {
        step(t);
    }
    glsl({
        state: CA_state[0].nearest,
        FP: `vec4(vec3(1.0 - state(UV).x)*0.5+0.25,1)`
    });
    requestAnimationFrame(render);
}

init();
