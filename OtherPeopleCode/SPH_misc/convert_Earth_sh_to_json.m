function convert_Earth_sh_to_json(shadr_filename,json_filename,sourceURL,sourceDescription,max_degree)


fid_in = fopen(shadr_filename,'r');

Rref = 6378.1363;
mu = 398600.4415;
norm = 1;

fid_out = fopen(json_filename,'w');
fprintf(fid_out,'{\n');
fprintf(fid_out,['    "BodyName":"Earth",\n']);
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
    data = textscan(line,'%d %d %f %f %f %f');
    if firstflag == 0
        fprintf(fid_out,'        },\n');
    elseif firstflag == 1
        firstflag = 0;
    end
    if data{1} > max_degree
        break
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