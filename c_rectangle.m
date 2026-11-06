function out = c_rectangle(a, ba, h, num_modes, elems_per_side)
% C_RECTANGLE  Clamped rectangular plate, full 3D model.
%
% Builds, meshes, solves and EXPORTS the normalized mode shapes
%     W = w/N,   N = sqrt(intop1(w*w))          (same definition as unit_sq.m)
% along two mid-lines and the z = 0 mid-plane:
%     outputs_modes/em_ba<b>_a<a>_h<h>_X.csv   y=0, z=0, x in [-a, +a]
%     outputs_modes/em_ba<b>_a<a>_h<h>_Y.csv   x=0, z=0, y in [-ba*a, +ba*a]
%     outputs_modes/em_ba<b>_a<a>_h<h>_W.csv   xy plane at z=0, nx*ny grid
%
% One CSV column per eigenmode index (header W1, W2, ...), all taken from the
% same solution record with with(k,W) + looplevelinput {'first'}.
%
% export_idx : (optional) eigenmode indices to export, default 1:num_modes.
%              Pass e.g. [1 2 3 ... 13] to keep only the modes that appear in
%              comsol_modes.csv for this aspect ratio.

import com.comsol.model.*
import com.comsol.model.util.*

export_idx = 1:num_modes;


npts_line  = 400;              % points along the X and Y cuts
npts_plane = 400;              % nx = ny on the mid-plane
out_dir    = 'outputs_modes';  % relative to the working directory

if ~exist(out_dir, 'dir'), mkdir(out_dir); end

model = ModelUtil.create('Model');

model.modelPath('/media/alec/DATA/PROJEKTY/plate_eigenfunctions/comsol_files');
model.label('eigenfunctions_rectangle.mph');

model.param.set('a', [num2str(a), '[m]']);
model.param.set('ba', [num2str(ba)]);
model.param.set('H', [num2str(h), '[m]']);
model.param.set('elems_per_side', [num2str(elems_per_side)]);

% Create and Reference Component/Geometry
comp1 = model.component.create('comp1', false);
geom1 = comp1.geom.create('geom1', 3);
comp1.curvedInterior(false);
geom1.geomRep('comsol');

% Create Block  (plate spans -a..a, -b*a..b*a, -H/2..H/2 -> mid-plane is z=0)
blk1 = geom1.create('blk1', 'Block');
blk1.set('size', {'2*a', '2*ba*a', 'H'});
blk1.set('base', 'center');

% Create WorkPlane and nested geometry
wp1 = geom1.create('wp1', 'WorkPlane');
wp1.active(false);
wp1.set('quickz', 'H/2');
wp1.set('displ', {'-a/2', '-a/2'});
wp1.set('unite', true);

geom1.run;

% Reference the component
comp1 = model.component('comp1');

var1 = comp1.variable.create('var1');
var1.set('N', 'sqrt(intop1(w*w))');
var1.set('W', 'w/N');

mat1 = comp1.material.create('mat1', 'Common');
mat1.label('Silicon');
mat1.selection.set(1); % Assign to Domain 1
mat1.propertyGroup.create('def', 'Basic');
mat1.propertyGroup('def').set('density', '2329[kg/m^3]');
mat1.propertyGroup.create('Enu', 'Enu', 'Young''s modulus and Poisson''s ratio');
mat1.propertyGroup('Enu').set('E', '170e9[Pa]');
mat1.propertyGroup('Enu').set('nu', '0.28');

% coupling operators for variables
comp1.cpl.create('intop1', 'Integration');
comp1.cpl.create('maxop1', 'Maximum');
comp1.cpl('intop1').selection.geom('geom1', 2);
comp1.cpl('intop1').selection.set([3]);
comp1.cpl('maxop1').selection.set([1]);

model.thermodynamics.label('Thermodynamics Package');

comp1.physics.create('solid', 'SolidMechanics', 'geom1');
comp1.physics('solid').selection.set([1]);
comp1.physics('solid').create('fix1', 'Fixed', 2);
comp1.physics('solid').feature('fix1').selection.set([1 2 5 6]);

comp1.physics('solid').prop('AdvancedSettings').set('GroupPhysOdesRc', false);
comp1.physics('solid').prop('AdvancedSettings').set('GroupPhysOdesAtt', false);
comp1.physics('solid').prop('AdvancedSettings').set('GroupPhysOdesRd', false);
comp1.physics('solid').selection.set([1]);

comp1.physics('solid').feature('lemm1').set('minput_temperature_src', 'userdef');
comp1.physics('solid').feature('lemm1').set('minput_strainreferencetemperature_src', 'userdef');
comp1.physics('solid').feature('dcnt1').set('pairDisconnect', true);
comp1.physics('solid').feature('dcnt1').label('Contact');
comp1.physics('solid').feature('dcont1').set('pairDisconnect', true);
comp1.physics('solid').feature('dcont1').label('Continuity');
comp1.physics('solid').feature('fix1').set('constraintMethod', 'elemental');

% mesh
mesh1 = comp1.mesh.create('mesh1');
mesh1.feature('size').set('custom', 'on');
mesh1.feature('size').set('hmax', '2*min(a,ba*a)/elems_per_side');
mesh1.feature('size').set('hmin', 1e-7);
mesh1.create('ftri1', 'FreeTri');
mesh1.create('dis1', 'Distribution');
mesh1.feature('dis1').set('numelem', 4);
mesh1.create('swe1', 'Sweep');
mesh1.feature('ftri1').selection.set([3]);
mesh1.feature('dis1').selection.set([1]);
mesh1.run;

% study definition
model.study.create('std1');
model.study('std1').create('eig', 'Eigenfrequency');
model.study('std1').feature('eig').set('neigs', num_modes);
model.study('std1').feature('eig').set('neigsactive', true);
model.study('std1').feature('eig').set('shift', '1000');
model.study('std1').run;

% ----------------------------------------------------------------------
% results nodes (all created AFTER the solve, so all modes are available)
% ----------------------------------------------------------------------

% dataset bound to the solution created by the study run
dset2 = model.result.dataset.create('dset2', 'Solution');
dset2.label('Solution 1');
dset2.set('solution', 'sol1');

% Results > Tables > Eigenfrequencies
tbl1 = model.result.table.create('tbl1', 'Table');
tbl1.label('Eigenfrequencies');

% Results > Eigenfrequencies > Global Evaluation
gev1 = model.result.numerical.create('gev1', 'EvalGlobal');
gev1.label('Global Evaluation');
gev1.set('data',  'dset2');
gev1.set('expr',  'freq');
gev1.set('unit',  'Hz');
gev1.set('descr', 'freq');
gev1.set('table', 'tbl1');

% Results > Mode Shape (W) > Surface 1
pg1 = model.result.create('pg1', 'PlotGroup3D');
pg1.label('Mode Shape (W)');
pg1.set('data', 'dset2');
surf1 = pg1.create('surf1', 'Surface');
surf1.label('Surface 1');
surf1.set('expr',  'W');
surf1.set('unit',  '1/m');
surf1.set('descr', 'Normalized displacement W');
surf1.create('def', 'Deform');

% ---- datasets for the exports -----------------------------------------
cln1 = model.result.dataset.create('cln1', 'CutLine3D');
cln1.label('Cut Line 3D X');
cln1.set('genpoints', [-a 0 0; a 0 0]);        % along x, at y = 0, z = 0

cln2 = model.result.dataset.create('cln2', 'CutLine3D');
cln2.label('Cut Line 3D Y');
cln2.set('genpoints', [0 -ba*a 0; 0 ba*a 0]);    % along y, at x = 0, z = 0

cpl2 = model.result.dataset.create('cpl2', 'CutPlane');
cpl2.label('Cut Plane W');
cpl2.set('quickplane', 'xy');
cpl2.set('quickz', '0');                       % z = 0 is the mid-plane

% ---- export expressions: one column per mode --------------------------
exprs = arrayfun(@(k) sprintf('with(%d,W)', k), export_idx, 'UniformOutput', false);
units = repmat({'1/m'}, 1, numel(export_idx));
descr = arrayfun(@(k) sprintf('W%d', k),        export_idx, 'UniformOutput', false);

base = sprintf('em_ba%1.2f_a%1.1e_h%1.1e', ba, a, h);

export_W(model, 'data2', 'X', 'cln1', exprs, units, descr, ...
         fullfile(out_dir, [base '_X.csv']), npts_line,  false);
export_W(model, 'data3', 'Y', 'cln2', exprs, units, descr, ...
         fullfile(out_dir, [base '_Y.csv']), npts_line,  false);
export_W(model, 'data4', 'W', 'cpl2', exprs, units, descr, ...
         fullfile(out_dir, [base '_W.csv']), npts_plane, true);

out = model;
end

% =======================================================================
function export_W(model, tag, lbl, dset, exprs, units, descr, fname, npts, isplane)
import com.comsol.model.*

ex = model.result.export.create(tag, 'Data');
ex.label(lbl);
ex.set('data', dset);
ex.set('looplevelinput', {'first'});   % mode selection comes from with(k,W)
ex.set('expr',  exprs);
ex.set('unit',  units);
ex.set('descr', descr);
ex.set('filename', fname);
ex.set('location', 'regulargrid');
if isplane
    ex.set('regulargridx2', npts);
    ex.set('regulargridy2', npts);
else
    ex.set('regulargridx1', npts);
end
ex.set('separator', ',');
ex.run;
end