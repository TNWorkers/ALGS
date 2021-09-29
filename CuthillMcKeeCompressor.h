#ifndef CUTHILLMCKEECOMPRESSOR
#define CUTHILLMCKEECOMPRESSOR

// suppresses compiler warnings:
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#define BOOST_ALLOW_DEPRECATED_HEADERS

#include <vector>
#include <iostream>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

template<typename Scalar>
Array<Scalar,Dynamic,Dynamic> compress_CuthillMcKee (const Array<Scalar,Dynamic,Dynamic> &A, bool PRINT=false)
{
	assert(A.rows() == A.cols());
	int N = A.rows();
	
	using namespace boost;
	typedef adjacency_list<vecS, vecS, undirectedS, property<vertex_color_t, default_color_type, property<vertex_degree_t,int> > > Graph;
	typedef graph_traits<Graph>::vertex_descriptor Vertex;
	typedef graph_traits<Graph>::vertices_size_type size_type;
	
	vector<pair<size_t, size_t>> edges;
	for (int i=0; i<N; ++i)
	for (int j=i+1; j<N; ++j)
	{
		if (abs(A(i,j)) > 0.) edges.push_back(pair<size_t, size_t>(i,j));
	}
	
	// construct graph
	Graph G(N);
	for (int i=0; i<edges.size(); ++i) add_edge(edges[i].first, edges[i].second, G);
	
	graph_traits<Graph>::vertex_iterator ui, ui_end;
	
	property_map<Graph,vertex_degree_t>::type deg = get(vertex_degree, G);
	for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui)
		deg[*ui] = degree(*ui, G);
	
	property_map<Graph, vertex_index_t>::type index_map = get(vertex_index, G);
	
	int b = bandwidth(G);
	
	vector<Vertex> inv_perm(num_vertices(G));
	vector<size_type> perm(num_vertices(G));
	
	vector<size_t> transform(N);
	Vertex s = vertex(0,G);
	cuthill_mckee_ordering(G, s, inv_perm.rbegin(), get(vertex_color, G), get(vertex_degree, G));
	int j=N-1;
	for (vector<Vertex>::const_iterator i=inv_perm.begin(); i!=inv_perm.end(); ++i)
	{
		transform[index_map[*i]] = j;
		--j;
	}
	
	if (PRINT)
	{
		for (int t=0; t<transform.size(); ++t)
		{
			lout << t << "→" << transform[t] << endl;
		}
	}
	
	for (size_type c = 0; c != inv_perm.size(); ++c)
	perm[index_map[inv_perm[c]]] = c;
	int b_new = bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]));
	
	Array<Scalar,Dynamic,Dynamic> res(N,N); res.setZero();
	for (int i=0; i<N; ++i)
	for (int j=0; j<i; ++j)
	{
		if (abs(A(i,j)) > 0.)
		{
			res(transform[i],transform[j]) = A(i,j);
			res(transform[j],transform[i]) = A(j,i);
		}
	}
	
	if (PRINT) lout << "Reverse Cuthill-McKee ordering starting at: " << s << ", bandwidth reduction: " << b << "→" << b_new << endl;
	
	return res;
}

vector<size_t> transform_CuthillMcKee (const ArrayXXd &A, bool PRINT=false)
{
	assert(A.rows() == A.cols());
	int N = A.rows();
	
	using namespace boost;
	typedef adjacency_list<vecS, vecS, undirectedS, property<vertex_color_t, default_color_type, property<vertex_degree_t,int> > > Graph;
	typedef graph_traits<Graph>::vertex_descriptor Vertex;
	typedef graph_traits<Graph>::vertices_size_type size_type;
	
	vector<pair<size_t, size_t>> edges;
	for (int i=0; i<N; ++i)
	for (int j=i+1; j<N; ++j)
	{
		if (abs(A(i,j)) > 0.) edges.push_back(pair<size_t, size_t>(i,j));
	}
	
	// construct graph
	Graph G(N);
	for (int i=0; i<edges.size(); ++i) add_edge(edges[i].first, edges[i].second, G);
	
	graph_traits<Graph>::vertex_iterator ui, ui_end;
	
	property_map<Graph,vertex_degree_t>::type deg = get(vertex_degree, G);
	for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui)
		deg[*ui] = degree(*ui, G);
	
	property_map<Graph, vertex_index_t>::type index_map = get(vertex_index, G);
	
	int b = bandwidth(G);
	
	vector<Vertex> inv_perm(num_vertices(G));
	vector<size_type> perm(num_vertices(G));
	
	vector<size_t> transform(N);
	Vertex s = vertex(0,G);
	cuthill_mckee_ordering(G, s, inv_perm.rbegin(), get(vertex_color, G), get(vertex_degree, G));
	int j=N-1;
	for (vector<Vertex>::const_iterator i=inv_perm.begin(); i!=inv_perm.end(); ++i)
	{
		transform[index_map[*i]] = j;
		--j;
	}
	
	if (PRINT)
	{
		for (int t=0; t<transform.size(); ++t)
		{
			lout << t << "→" << transform[t] << endl;
		}
	}
//	
//	for (size_type c = 0; c != inv_perm.size(); ++c)
//	perm[index_map[inv_perm[c]]] = c;
//	int b_new = bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]));
//	
//	ArrayXXd res(N,N); res.setZero();
//	for (int i=0; i<N; ++i)
//	for (int j=0; j<i; ++j)
//	{
//		if (abs(A(i,j)) > 0.)
//		{
//			res(transform[i],transform[j]) = A(i,j);
//			res(transform[j],transform[i]) = A(j,i);
//		}
//	}
//	
//	if (PRINT) lout << "Reverse Cuthill-McKee ordering starting at: " << s << ", bandwidth reduction: " << b << "→" << b_new << endl;
//	
//	return res;
	return transform;
}

#endif
