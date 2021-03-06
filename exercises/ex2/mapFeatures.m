function out = mapFeatures(X1, X2, degree)
	out = ones(size(X1(:,1)));%this also adds column of 1s for x_0
	
	for i=1:degree
		for j=0:i
			out(:, end+1) = (X1.^(i-j)).*(X2.^j);
		end
	end
end
