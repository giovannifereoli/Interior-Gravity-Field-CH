function convert_pds_shadr_to_json(bodyName,shadr_filename,json_filename,sourceURL,sourceDescription)


% fid = fopen('jggrx_1500e_sha.tab','r');
fid_in = fopen(shadr_filename,'r');

% Header line
line = fgetl(fid_in);
data = textscan(line,'%f, %f, %f, %d, %d, %d, %f, %f');

Rref = data{1};
mu = data{2};
max_degree = data{4};
max_order = data{5};
norm = data{6};

fid_out = fopen(json_filename,'w');
fprintf(fid_out,'{\n');
fprintf(fid_out,['    "BodyName":"' bodyName '",\n']);
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
while ~feof(fid_in)
    line = fgetl(fid_in);
    data = textscan(line,'%d,%d,%f,%f,%f,%f');
    if firstflag == 0
        fprintf(fid_out,'        },\n');
    elseif firstflag == 1
        firstflag = 0;
    end
    fprintf(fid_out,'        {\n');
	fprintf(fid_out,['            "degree":' num2str(data{1},'%d') ',\n']);
	fprintf(fid_out,['            "order":' num2str(data{2},'%d') ',\n']);
	fprintf(fid_out,['            "Cnm":' num2str(data{3},'%16.15E') ',\n']);
	fprintf(fid_out,['            "Snm":' num2str(data{4},'%16.15E') '\n']);
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
fclose(fid_in);

end