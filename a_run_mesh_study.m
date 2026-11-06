clear;

% ba_sweep = [1.03, 1.1, 1.18, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00];
% a_sweep = [3.0e-4, 6.0e-4, 9.0e-4, 1.2e-3, 1.5e-3];
% h_sweep = [4.0e-6, 7.0e-6, 1.0e-5, 1.3e-5, 1.6e-5];

ba_sweep = [1.03];
a_sweep = [3.0e-4];
h_sweep = [1.6e-5];

num_modes = 12;

% hmax = 2*min(a,b*a)/elems_per_side
elems_per_side_sweep = [41, 51,61,71,81,91,101,111,121];

results_data = zeros(length(ba_sweep)*length(a_sweep)*length(h_sweep)*length(elems_per_side_sweep), 4 + num_modes);

i=1;

ba=ba_sweep(1); a=a_sweep(1); h=h_sweep(1);
for elems_per_side = elems_per_side_sweep
    
    fprintf('Running: ba=%1.2f, a=%.2e, h=%.2e, elems_per_side=%d\n', ba, a, h, elems_per_side);

    filename_mph = sprintf('models_mesh_study/ef_full_ba%1.2f_a%1.1e_h%1.1e_%d.mph', ba, a, h, elems_per_side);

    % Build and Solve
    mobj = eigenfunctions_rectangle_full(a, ba, h, num_modes, elems_per_side);
    
    mphsave(mobj, filename_mph);
    d = dir(filename_mph);
    fprintf('Saved file: %s with %d bytes at timestamp %s\n', filename_mph, d.bytes, d.date);

    freqs = mphglobal(mobj, 'freq'); 
    
    % Store in the results matrix
    results_data(i, :) = [ba, a, h, elems_per_side, freqs'];
    i=i+1;


end

format short e;
disp(results_data)

headers = [{'ba'},{'a'},{'h'},{'elems_per_side'}, arrayfun(@(x) sprintf('f%d', x), 1:num_modes, 'UniformOutput', false)];
T_final = array2table(results_data, 'VariableNames', headers);


filename = 'mesh_study_v3.csv';
writetable(T_final, filename);
fprintf('Final results saved to %s\n', filename);



