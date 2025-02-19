function convert_EarthgravCVS_to_json(bodyName,csv_filename,json_filename,sourceURL,sourceDescription)


grav = readmatrix(csv_filename);

% Header line
Rref = 6378.1366;
mu = 3.9860043543609598E+05;
max_degree = 100;
max_order = 100;
norm = 0;

fid_out = fopen(json_filename,'w');
fprintf(fid_out,'{\n');
fprintf(fid_out,['    "BodyName": "' bodyName '",\n']);
fprintf(fid_out,['    "ReferenceRadius":' num2str(Rref,6) ',\n']);
fprintf(fid_out,['    "GravitationalParameter":' num2str(mu,12) ',\n']);
if norm == 0
    fprintf(fid_out,'    "Normalized":false,\n');
elseif norm == 1
    fprintf(fid_out,'    "Normalized":true,\n');
else
    error('Unknown normalization! Panic!')
end
fprintf(fid_out,['    "MaxDegree":' num2str(max_degree,'%d') ',\n']);
fprintf(fid_out,'    "Coefficients":[\n');

firstflag = 1;
for ii = 1:size(grav,1)
    if firstflag == 0
        fprintf(fid_out,'        },\n');
    elseif firstflag == 1
        firstflag = 0;
    end
    fprintf(fid_out,'        {\n');
	fprintf(fid_out,['            "degree":' num2str(grav(ii,1),'%d') ',\n']);
	fprintf(fid_out,['            "order":' num2str(grav(ii,2),'%d') ',\n']);
	fprintf(fid_out,['            "Cnm":' num2str(grav(ii,3),'%16.15E') ',\n']);
	fprintf(fid_out,['            "Snm":' num2str(grav(ii,4),'%16.15E') '\n']);
end
fprintf(fid_out,'        }\n');
fprintf(fid_out,'    ],\n');
fprintf(fid_out,'    "source":[\n');
fprintf(fid_out,'        {\n');
fprintf(fid_out,['            "url":"' sourceURL '",\n']);
fprintf(fid_out,['            "desciption":"' sourceDescription '"\n']);
fprintf(fid_out,'        }\n');
fprintf(fid_out,'    ]\n');
fprintf(fid_out,'}');

fclose(fid_out);


end