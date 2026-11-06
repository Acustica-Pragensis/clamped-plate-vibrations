% Eigenfrequencies of the fully clamped square plate for the four materials of
% silicon, aluminium, iron, copper.
%
% The model is the one of c_rectangle.m (3D Solid Mechanics, quadratic elements,
% meshed as in Appendix A). It is built ONCE and re-solved for each material,
% because only the material properties enter the eigenvalue problem.
%
% Geometry used for the comparison: a = b = 900 um (ba = b/a = 1), h = 10 um.
%
% Output: ef_materials.csv  ->  material, E_GPa, nu, rho_kgm3, f1..f17 [Hz]
%         (the computed modes are sorted by frequency, as returned by the
%          eigensolver -- see the note at the end of the file)

clear; clc;

% ------------------------------------------------------------------ config
a   = 900e-6;          % smaller half-side [m]
ba  = 1.0;             % b/a  (1 -> square plate, a = b)
h   = 10e-6;           % thickness [m]

num_modes      = 17;   % computed modes, enough to reach (3,3)
elems_per_side = 61;   % N from Appendix A

OUT_CSV  = 'ef_materials.csv';
SAVE_MPH = false;      % true -> keep one .mph per material

% material properties (E, nu, rho) for the four materials
mat_name = {'silicon',   'aluminium', 'iron',    'copper'};
mat_E    = [170e9,        70e9,        200e9,     110e9 ];
mat_nu   = [0.28,         0.33,        0.29,      0.35  ];
mat_rho  = [2329,         2700,        7870,      8960  ];

n_mat = numel(mat_name);

% If you run against an external COMSOL server, connect first:
%   mphstart(2036);

% ------------------------------------------------ model (built only once)
fprintf('building geometry, physics and mesh ... ');
t0 = tic;
model = plate_model(a, ba, h, num_modes, elems_per_side);
fprintf('done in %.1f s\n\n', toc(t0));

% ------------------------------------------------------- solve per material
results = nan(n_mat, num_modes);

for k = 1:n_mat

    fprintf('[%d/%d] %-10s E = %6.0f GPa  nu = %.2f  rho = %4.0f kg/m3 ... ', ...
            k, n_mat, mat_name{k}, mat_E(k)/1e9, mat_nu(k), mat_rho(k));
    t = tic;

    model = set_material(model, mat_E(k), mat_nu(k), mat_rho(k));
    model.study('std1').run;

    freqs = mphglobal(model, 'freq');      % Hz
    freqs = freqs(:)';

    nf = min(numel(freqs), num_modes);
    results(k, 1:nf) = freqs(1:nf);
    if nf < num_modes
        warning('%s: only %d modes returned instead of %d', ...
                mat_name{k}, nf, num_modes);
    end

    fprintf('%.1f s   f1 = %9.2f Hz\n', toc(t), results(k, 1));

    if SAVE_MPH
        mphsave(model, sprintf('ef_materials_%s.mph', mat_name{k}));
    end

end

% --------------------------------------------------------------- save CSV
T = table(mat_name(:), mat_E(:)/1e9, mat_nu(:), mat_rho(:), ...
          'VariableNames', {'material', 'E_GPa', 'nu', 'rho_kgm3'});
for j = 1:num_modes
    T.(sprintf('f%d', j)) = results(:, j);
end
writetable(T, OUT_CSV);
fprintf('\nwrote %s\n', OUT_CSV);

% ------------------------------------------------------------ print table
hdr = sprintf('%12s', 'material');
for j = 1:num_modes
    hdr = [hdr sprintf('%10s', sprintf('f%d', j))];
end
fprintf('\ncomputed eigenfrequencies [Hz]\n');
disp(hdr);
for k = 1:n_mat
    row = sprintf('%12s', mat_name{k});
    for j = 1:num_modes
        row = [row sprintf('%10.1f', results(k, j))];
    end
    disp(row);
end

% ======================================================================
%  MODEL
% ======================================================================
function model = plate_model(a, ba, h, num_modes, elems_per_side)

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');
model.modelPath(pwd);
model.label('ef_materials.mph');

model.param.set('a', [num2str(a), '[m]']);
model.param.set('ba', num2str(ba));
model.param.set('H',  [num2str(h), '[m]']);
model.param.set('elems_per_side', num2str(elems_per_side));

% ---------------------------------------------------------------- geometry
comp1 = model.component.create('comp1', false);
geom1 = comp1.geom.create('geom1', 3);
comp1.curvedInterior(false);
geom1.geomRep('comsol');

blk1 = geom1.create('blk1', 'Block');
blk1.set('size', {'2*a', '2*ba*a', 'H'});
blk1.set('base', 'center');
geom1.run;

comp1 = model.component('comp1');

% ------------------------------------------------- variables and material
var1 = comp1.variable.create('var1');
var1.set('N', 'sqrt(intop1(w*w))');
var1.set('W', 'w/N');

% placeholder material, replaced in set_material()
mat1 = comp1.material.create('mat1', 'Common');
mat1.label('Material');
mat1.selection.set(1);
mat1.propertyGroup.create('def', 'Basic');
mat1.propertyGroup('def').set('density', '2329[kg/m^3]');
mat1.propertyGroup.create('Enu', 'Enu', 'Young''s modulus and Poisson''s ratio');
mat1.propertyGroup('Enu').set('E', '170e9[Pa]');
mat1.propertyGroup('Enu').set('nu', '0.28');

comp1.cpl.create('intop1', 'Integration');
comp1.cpl('intop1').selection.geom('geom1', 2);
comp1.cpl('intop1').selection.set([3]);

% ---------------------------------------------------------------- physics
comp1.physics.create('solid', 'SolidMechanics', 'geom1');
comp1.physics('solid').selection.set([1]);
comp1.physics('solid').create('fix1', 'Fixed', 2);
comp1.physics('solid').feature('fix1').selection.set([1 2 5 6]);

comp1.physics('solid').prop('AdvancedSettings').set('GroupPhysOdesRc',  false);
comp1.physics('solid').prop('AdvancedSettings').set('GroupPhysOdesAtt', false);
comp1.physics('solid').prop('AdvancedSettings').set('GroupPhysOdesRd',  false);

comp1.physics('solid').feature('lemm1').set('minput_temperature_src', 'userdef');
comp1.physics('solid').feature('lemm1').set('minput_strainreferencetemperature_src', 'userdef');

comp1.physics('solid').feature('dcnt1').set('pairDisconnect', true);
comp1.physics('solid').feature('dcnt1').label('Contact');
comp1.physics('solid').feature('dcont1').set('pairDisconnect', true);
comp1.physics('solid').feature('dcont1').label('Continuity');

comp1.physics('solid').feature('fix1').set('constraintMethod', 'elemental');

% ------------------------------------------------------------------- mesh
mesh1 = comp1.mesh.create('mesh1');
mesh1.feature('size').set('custom', 'on');
mesh1.feature('size').set('hmax', '2*min(a,ba*a)/elems_per_side');
mesh1.feature('size').set('hmin', 1e-7);
mesh1.create('ftri1', 'FreeTri');
mesh1.create('dis1',  'Distribution');
mesh1.feature('dis1').set('numelem', 4);
mesh1.create('swe1', 'Sweep');
mesh1.feature('ftri1').selection.set([3]);
mesh1.feature('dis1').selection.set([1]);
mesh1.run;

% ------------------------------------------------------------------ study
model.study.create('std1');
model.study('std1').create('eig', 'Eigenfrequency');
model.study('std1').feature('eig').set('neigs', num_modes);
model.study('std1').feature('eig').set('neigsactive', true);
model.study('std1').feature('eig').set('shift', '1000');
end

% ======================================================================
function model = set_material(model, E, nu, rho)
mat1 = model.component('comp1').material('mat1');
mat1.propertyGroup('def').set('density', sprintf('%.10g[kg/m^3]', rho));
mat1.propertyGroup('Enu').set('E',  sprintf('%.10g[Pa]', E));
mat1.propertyGroup('Enu').set('nu', sprintf('%.10g', nu));
end