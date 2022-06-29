### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a3959e55-294b-47b1-a869-1ca33d6e8089
using DelimitedFiles

# ╔═╡ 979f6b39-199b-434a-8c1c-f2bd9dfce2d1
using Statistics

# ╔═╡ 51004be1-c490-4669-8664-3e5d42059afb
md"""
# Naive Bayes Models
"""

# ╔═╡ f11098da-f61e-11ec-3264-f7afa484a1b8
PATH=string(@__DIR__) * "/dataset/"

# ╔═╡ 071016a3-ccbe-413c-a501-855ec5e1bdc3
"""
	Wisconsin Diagnostic Breast Cancer (ID number, diagnostic, features..)
	 * INPUT: Path of file
	 * OUTPUT: Features matrix X and Diagnostic Result vector y
"""
function readData(filename)
	buff = readdlm(PATH * filename, ',') # read data from filename in PATH, separated by comma
	X = buff[:, 3:end] # get data from column 3 to the end (features)
	y = Int.(buff[:,2]) .+ 1 # get diagnostic results, normalize from (0, 1) -> (1, 2)
	return (X, y)
end

# ╔═╡ 28a4b268-83eb-45da-9dcc-3669048eade6
md"""
## Implementations
"""

# ╔═╡ 4e675417-d803-4e88-aa10-02781a16d441
"""
	INPUT: Training data (X, y). X is a matrix of shape N x (D + 1), y is a column vector of length N.
	OUTPUT: Tuple (μ, σ, θ).
"""
function train(X, y)
	K = length(unique(y)) # number of characteristic values of x (in this case = 2)
	N, D = size(X) # get the number of tuples (N) and the number of features (D) of each tuple
	μ = zeros(D, K)
	σ = zeros(D, K)
	θ = zeros(K)
	for k=1:K
		idk = (y .== k)
		μ[:,k] = vec(mean(X[idk, :], dims = 1))
		σ[:,k] = vec(std(X[idk, :], dims = 1))
		θ[k] = sum(idk)/N
	end
	return (μ, σ, θ)
end

# ╔═╡ d24cadfa-80dc-4352-bf75-7c922d945dec
"""
	INPUT: Test Data xNew
	OUTPUT: Result of type prediction of xNew
"""
function classify(μ, σ, θ, xNew)
	D, K = size(μ)
	scores = zeros(K)
	for k=1:K
		s = 0
		for j=1:D
			s = s - log(σ[j,k]) - (xNew[j] - μ[j,k])^2/(2*σ[j,k]^2)
		end
		scores[k] = log(θ[k]) + s
	end

	return argmax(scores)
end

# ╔═╡ ce7671c3-a69d-4585-92fc-6df80b3ad135
"""
	Check the fit of the model with the actual result
"""
function evaluate(μ, σ, θ, X, y)
	N = length(y)
	z = map(i -> classify(μ, σ, θ, X[i, :]), 1:N)
	sum(z .== y) / N
end

# ╔═╡ 3212acaa-e481-402a-a0eb-e66ad6e13f95
md"""
## Testing on Breast Cancer Diagnosis Dataset
"""

# ╔═╡ 850b363c-9e09-4d9e-a894-edad8c9280b6
# Read input data, remove the first column (ID number): X is the features
# (columns 3 - 32), y is diagnosis [1, 2] with 1 being malignant (malignant),
# 2 being benign (benign)
X, y = readData("wdbc.txt")

# ╔═╡ cf88e8c7-e8ce-4bde-b12a-748f59beae53
# Get the index of rows with y = 1 (malignant)
id1 = (y .== 1)

# ╔═╡ 4c5f799d-f614-4bd3-805d-1ca7cfca5d72
# Retrieve rows with y = 1 (malignant)
X_m = X[id1, :]

# ╔═╡ cdffb051-49be-4f78-8f74-e7087ac3519b
# Get the index of rows with y = 2 (benign)
id2 = (y .== 2)

# ╔═╡ 02d99408-d2e4-4d03-86c9-fe654de0a8c6
# Retrieve rows with y = 2 (benign)
X_b = X[id2, :]

# ╔═╡ 63e318e9-76c3-439b-b407-2d8c2f51cdf2
# Stretch matrix X_b to vector, average features on benign data
vec(mean(X_b, dims=1))

# ╔═╡ db0cdf5e-d687-4d0a-ab6d-03dcc8269920
# Model training
μ, σ, θ = train(X, y)

# ╔═╡ 424215bd-b3bc-4cf0-b048-fa7a57280c5b
# Try on the model obtained with the 100th element X
classify(μ, σ, θ, X[100,:])

# ╔═╡ a361034d-9b54-4e30-8542-7e4e9e0caec9
# Evaluate the accuracy of the model
evaluate(μ, σ, θ, X, y)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
"""

# ╔═╡ Cell order:
# ╟─51004be1-c490-4669-8664-3e5d42059afb
# ╠═f11098da-f61e-11ec-3264-f7afa484a1b8
# ╠═a3959e55-294b-47b1-a869-1ca33d6e8089
# ╠═979f6b39-199b-434a-8c1c-f2bd9dfce2d1
# ╠═071016a3-ccbe-413c-a501-855ec5e1bdc3
# ╟─28a4b268-83eb-45da-9dcc-3669048eade6
# ╠═4e675417-d803-4e88-aa10-02781a16d441
# ╠═d24cadfa-80dc-4352-bf75-7c922d945dec
# ╠═ce7671c3-a69d-4585-92fc-6df80b3ad135
# ╟─3212acaa-e481-402a-a0eb-e66ad6e13f95
# ╠═850b363c-9e09-4d9e-a894-edad8c9280b6
# ╠═cf88e8c7-e8ce-4bde-b12a-748f59beae53
# ╠═4c5f799d-f614-4bd3-805d-1ca7cfca5d72
# ╠═cdffb051-49be-4f78-8f74-e7087ac3519b
# ╠═02d99408-d2e4-4d03-86c9-fe654de0a8c6
# ╠═63e318e9-76c3-439b-b407-2d8c2f51cdf2
# ╠═db0cdf5e-d687-4d0a-ab6d-03dcc8269920
# ╠═424215bd-b3bc-4cf0-b048-fa7a57280c5b
# ╠═a361034d-9b54-4e30-8542-7e4e9e0caec9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
