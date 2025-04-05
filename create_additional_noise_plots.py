from io import StringIO
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

## for each dnn2a,dnn2b..., we create a single plot out of input, output, variance based noise
## we should save 8 plots total
# noise_experiments = {
#     'v43': {
#         'dnn2a':{
#             'input':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.570635	0.569727	0.569766	0.569988	0.570278	0.570386	0.570318	0.570314
# 0.5	0.569194	0.564842	0.56526	0.567503	0.56935	0.570048	0.570294	0.570058
# 1	0.567581	0.558076	0.558742	0.563719	0.567596	0.569496	0.570345	0.569602
# 1.5	0.566384	0.552149	0.552045	0.560493	0.56635	0.569133	0.570324	0.569201
# 2	0.564685	0.544583	0.545275	0.556611	0.564573	0.568148	0.570169	0.568268
# 2.5	0.563061	0.537411	0.538278	0.552129	0.562555	0.568198	0.570123	0.567193"""),
#             'output':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.569841	0.569602	0.569934	0.570032	0.570164	0.570281	0.57036	0.555954
# 0.5	0.566072	0.565125	0.565203	0.567667	0.569315	0.570106	0.570279	0.495522
# 1	0.561468	0.558278	0.558889	0.563744	0.567793	0.569398	0.570245	0.421172
# 1.5	0.55661	0.551324	0.552229	0.56015	0.56652	0.56902	0.570259	0.372887
# 2	0.552801	0.545349	0.54653	0.556436	0.564705	0.568288	0.570085	0.340294
# 2.5	0.546953	0.537796	0.538353	0.55184	0.562696	0.56786	0.570008	0.309926"""),
#             'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.56936	0.566996	0.567417	0.568754	0.569869	0.570288	0.570337	0.570309
# 0.5	0.563925	0.550788	0.552044	0.56013	0.566256	0.568957	0.570297	0.569007
# 1	0.556053	0.529175	0.530234	0.548351	0.560688	0.567554	0.569786	0.566507
# 1.5	0.54817	0.504271	0.506603	0.533172	0.554224	0.565641	0.569231	0.565427
# 2	0.540021	0.479744	0.48401	0.518721	0.547987	0.563428	0.569046	0.563001
# 2.5	0.531556	0.455687	0.458002	0.502388	0.539693	0.561269	0.568477	0.560512"""),
#         },
#         'dnn2b':{
#             'input':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.57022	0.570294	0.570651	0.570301	0.570144	0.570325	0.57016	0.570185
# 0.5	0.570404	0.568701	0.570061	0.570421	0.570328	0.570254	0.570366	0.570305
# 1	0.569792	0.566194	0.568843	0.570226	0.570554	0.570392	0.570344	0.569942
# 1.5	0.56901	0.564067	0.567679	0.568784	0.570512	0.570162	0.570412	0.569728
# 2	0.568645	0.561736	0.566456	0.569596	0.570601	0.570308	0.570419	0.569131
# 2.5	0.56819	0.558954	0.56534	0.569153	0.570748	0.570405	0.57042	0.569138"""),
#             'output':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.570236	0.570226	0.570639	0.570528	0.570238	0.570252	0.570246	0.563077
# 0.5	0.569026	0.568893	0.569975	0.570485	0.570332	0.570282	0.57027	0.531998
# 1	0.567562	0.566272	0.568606	0.570265	0.570622	0.570318	0.570343	0.475114
# 1.5	0.566065	0.563766	0.567714	0.569815	0.570716	0.570351	0.570315	0.432086
# 2	0.564695	0.56221	0.566518	0.569394	0.570705	0.570597	0.570537	0.389586
# 2.5	0.563889	0.558719	0.56531	0.569146	0.570695	0.570413	0.570425	0.371169"""),
#             'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.569507	0.567571	0.56966	0.570321	0.570229	0.570194	0.570274	0.570265
# 0.5	0.563787	0.555888	0.563846	0.569122	0.570727	0.570359	0.570406	0.56975
# 1	0.558005	0.538883	0.557391	0.567464	0.570792	0.570529	0.570525	0.568419
# 1.5	0.551364	0.519324	0.547953	0.565513	0.570406	0.570522	0.570589	0.567282
# 2	0.543982	0.501467	0.539818	0.563069	0.569929	0.570879	0.570588	0.566407
# 2.5	0.537422	0.482292	0.531675	0.561288	0.569449	0.570703	0.570705	0.565016"""),
#         },
#         'dnn2c':{
#             'input':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8	layer_9
# 0.1	0.563533	0.561444	0.560865	0.559926	0.560019	0.560354	0.56112	0.562395	0.563145
# 0.5	0.561274	0.55527	0.551965	0.549086	0.549441	0.551569	0.554792	0.559024	0.560231
# 1	0.559301	0.54747	0.541666	0.534404	0.536449	0.538554	0.547098	0.554865	0.558355
# 1.5	0.558118	0.538917	0.530901	0.521151	0.524295	0.528561	0.539073	0.549862	0.556236
# 2	0.556801	0.531209	0.519559	0.506944	0.51035	0.516057	0.530156	0.545667	0.553794
# 2.5	0.555255	0.523561	0.507931	0.493434	0.49698	0.503488	0.522395	0.542351	0.551571"""),
#             'output':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8	layer_9
# 0.1	0.562255	0.56088	0.559893	0.559139	0.558867	0.559535	0.560278	0.562303	0.558525
# 0.5	0.558278	0.552265	0.546954	0.544298	0.542858	0.5475	0.551085	0.556971	0.540298
# 1	0.553828	0.541723	0.530027	0.525571	0.521843	0.532139	0.539011	0.55099	0.514987
# 1.5	0.54971	0.53157	0.51476	0.508087	0.503943	0.515104	0.526821	0.546833	0.488602
# 2	0.544911	0.520128	0.498826	0.490544	0.484373	0.50209	0.51477	0.541112	0.465835
# 2.5	0.539972	0.508816	0.483863	0.472946	0.467599	0.487607	0.500543	0.533908	0.436918"""),
#             'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8	layer_9
# 0.1	0.56151	0.55821	0.556455	0.555474	0.555741	0.557119	0.559003	0.560891	0.560745
# 0.5	0.554872	0.536436	0.52966	0.524478	0.528201	0.536411	0.543975	0.554467	0.552409
# 1	0.546882	0.509216	0.496929	0.489318	0.496486	0.507271	0.525489	0.545995	0.54188
# 1.5	0.538399	0.483591	0.469796	0.456738	0.465734	0.481083	0.506148	0.536721	0.53057
# 2	0.527907	0.45671	0.442049	0.42244	0.438946	0.45746	0.487794	0.525341	0.516979
# 2.5	0.519935	0.432758	0.417785	0.397756	0.413165	0.436699	0.470186	0.517168	0.508114"""),
#         },
#         'dnn2d':{
#             'input':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8	layer_9
# 0.1	0.561486	0.559771	0.55873	0.557886	0.557957	0.558814	0.559883	0.560864	0.561206
# 0.5	0.559897	0.552929	0.549902	0.545105	0.546968	0.54894	0.553825	0.557646	0.559315
# 1	0.557811	0.544997	0.539157	0.53052	0.531392	0.536073	0.545991	0.554087	0.557008
# 1.5	0.556178	0.537043	0.52748	0.514341	0.516184	0.522168	0.53729	0.549062	0.554526
# 2	0.554848	0.528961	0.516759	0.498889	0.502315	0.507863	0.528889	0.544917	0.553298
# 2.5	0.553264	0.519284	0.503798	0.483499	0.486064	0.494125	0.520035	0.539955	0.551158"""),
#             'output':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8	layer_9
# 0.1	0.560865	0.559268	0.557697	0.556943	0.556356	0.558079	0.559352	0.560699	0.557055
# 0.5	0.556904	0.550405	0.544354	0.539274	0.539091	0.544053	0.550308	0.556116	0.541037
# 1	0.550986	0.538867	0.527771	0.520938	0.516254	0.526373	0.537266	0.550569	0.512211
# 1.5	0.546535	0.528692	0.511269	0.500113	0.494859	0.509095	0.523064	0.544203	0.485555
# 2	0.540108	0.516993	0.494444	0.478986	0.472638	0.490138	0.510652	0.539334	0.460414
# 2.5	0.535148	0.506114	0.478973	0.460068	0.453212	0.473189	0.49661	0.531562	0.432683"""),
#             'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8	layer_9
# 0.1	0.560232	0.555726	0.554136	0.552794	0.553587	0.555161	0.557996	0.55986	0.559719
# 0.5	0.553702	0.534769	0.528062	0.520603	0.523272	0.531421	0.544009	0.552361	0.551777
# 1	0.543952	0.50895	0.496513	0.478256	0.484748	0.498077	0.524112	0.543904	0.542009
# 1.5	0.535152	0.484452	0.464662	0.437508	0.45146	0.470022	0.503922	0.535091	0.530149
# 2	0.524355	0.462814	0.440988	0.404715	0.424012	0.43882	0.482805	0.522839	0.515336
# 2.5	0.517119	0.440164	0.418807	0.375599	0.39551	0.418296	0.460313	0.51389	0.502425"""),
#         },
#     },
#     'v44': {
#         'dnn2a':{
#             'input':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.554919	0.553976	0.554455	0.555319	0.555994	0.556141	0.556312	0.555703
# 0.5	0.554359	0.550819	0.553134	0.554835	0.555118	0.555451	0.555739	0.554804
# 1	0.553335	0.545766	0.552142	0.554628	0.555948	0.555102	0.55542	0.554597
# 1.5	0.552226	0.540751	0.550193	0.553888	0.554577	0.554885	0.555251	0.554237
# 2	0.550986	0.535923	0.549079	0.553277	0.554492	0.554698	0.555073	0.554335
# 2.5	0.550415	0.530693	0.547141	0.553157	0.554479	0.554792	0.554976	0.553824"""),
#             'output':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.554447	0.55437	0.554638	0.555312	0.556065	0.556169	0.556222	0.550173
# 0.5	0.551848	0.550789	0.553266	0.55458	0.555203	0.555486	0.555711	0.495067
# 1	0.548776	0.546122	0.551986	0.554215	0.55481	0.555119	0.555425	0.426927
# 1.5	0.545834	0.540965	0.550401	0.553961	0.554706	0.554737	0.555229	0.383546
# 2	0.542333	0.536234	0.548892	0.553572	0.554577	0.554698	0.554984	0.337038
# 2.5	0.537497	0.531126	0.546806	0.553467	0.554365	0.554462	0.554967	0.317924"""),
#             'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.553821	0.552101	0.55392	0.554803	0.555438	0.55585	0.556128	0.5554
# 0.5	0.549172	0.539199	0.549519	0.553897	0.554605	0.55501	0.555374	0.55476
# 1	0.54115	0.520763	0.543517	0.553163	0.554021	0.554573	0.554956	0.55423
# 1.5	0.533114	0.502291	0.537528	0.55157	0.553938	0.554261	0.554787	0.553922
# 2	0.526025	0.485496	0.531045	0.550175	0.553577	0.554492	0.554508	0.55313
# 2.5	0.518196	0.46659	0.523511	0.548339	0.553071	0.55393	0.554513	0.553001"""),
#         },
#         'dnn2b':{
#             'input':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.563025	0.561294	0.562677	0.563355	0.563724	0.563826	0.563844	0.563642
# 0.5	0.561148	0.556251	0.560254	0.562233	0.563284	0.563554	0.563615	0.562859
# 1	0.559668	0.54992	0.558678	0.561338	0.56285	0.563182	0.563459	0.562526
# 1.5	0.55783	0.543669	0.556063	0.560553	0.562554	0.562883	0.563278	0.561977
# 2	0.556207	0.537116	0.554037	0.560299	0.562167	0.562842	0.562969	0.561648
# 2.5	0.554993	0.532193	0.551612	0.559705	0.562118	0.562723	0.562803	0.561235"""),
#             'output':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.561922	0.561221	0.562704	0.563245	0.563686	0.56376	0.563809	0.555095
# 0.5	0.557616	0.55613	0.560537	0.562425	0.563204	0.563519	0.563624	0.506059
# 1	0.5535	0.549806	0.558281	0.561583	0.562884	0.563345	0.563397	0.426475
# 1.5	0.548762	0.544205	0.556037	0.56071	0.562624	0.563014	0.563244	0.367713
# 2	0.544174	0.538525	0.554235	0.560041	0.56221	0.562918	0.563273	0.333916
# 2.5	0.539671	0.531806	0.551484	0.559306	0.561897	0.562649	0.563056	0.310051"""),
#             'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.560742	0.558425	0.561443	0.562663	0.563433	0.563647	0.563712	0.563373
# 0.5	0.553504	0.540926	0.556065	0.560849	0.56272	0.563051	0.563444	0.562292
# 1	0.544207	0.521919	0.548251	0.558812	0.561907	0.562715	0.56296	0.561427
# 1.5	0.535511	0.502827	0.541066	0.55672	0.561331	0.562259	0.562708	0.560347
# 2	0.526893	0.483825	0.533242	0.554852	0.560689	0.561718	0.562424	0.559651
# 2.5	0.517261	0.469191	0.526678	0.552427	0.560063	0.561396	0.5622	0.559217"""),
#         },
#         'dnn2c':{
#             'input':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.5672	0.566581	0.566782	0.566722	0.56723	0.567278	0.567191	0.567472
# 0.5	0.566508	0.561225	0.562188	0.562679	0.564601	0.56551	0.566593	0.56679
# 1	0.565361	0.554911	0.556109	0.557835	0.561409	0.563204	0.565347	0.566243
# 1.5	0.563853	0.548267	0.550076	0.552302	0.557389	0.560996	0.56385	0.565248
# 2	0.562341	0.543327	0.544947	0.548229	0.554131	0.558899	0.562469	0.56484
# 2.5	0.561559	0.536816	0.538577	0.543383	0.549291	0.556023	0.561009	0.564069"""),
#             'output':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.567009	0.566107	0.566062	0.566621	0.566828	0.566993	0.567154	0.565645
# 0.5	0.563939	0.559385	0.558925	0.560973	0.562939	0.565314	0.566117	0.553732
# 1	0.559763	0.550574	0.550521	0.554446	0.556896	0.562301	0.563929	0.52488
# 1.5	0.555484	0.542475	0.541879	0.548037	0.551448	0.558814	0.561582	0.499279
# 2	0.551644	0.534492	0.532571	0.541106	0.545865	0.555212	0.559174	0.461127
# 2.5	0.547674	0.526528	0.523803	0.533769	0.539376	0.550261	0.555484	0.429689"""),
#             'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.566094	0.563498	0.564459	0.564904	0.56622	0.566624	0.567131	0.567166
# 0.5	0.559575	0.545511	0.549619	0.554027	0.5595	0.562889	0.565393	0.565155
# 1	0.550726	0.524142	0.53098	0.540683	0.550558	0.557574	0.562404	0.562511
# 1.5	0.542311	0.506237	0.513728	0.526352	0.539975	0.550882	0.558226	0.559051
# 2	0.533579	0.488627	0.498647	0.513467	0.530118	0.543669	0.554353	0.554878
# 2.5	0.525632	0.471433	0.481532	0.500175	0.518604	0.536996	0.55017	0.549997"""),
#         },
#         'dnn2d':{
#             'input':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.558929	0.557952	0.557792	0.557764	0.558338	0.558861	0.558907	0.559058
# 0.5	0.557438	0.55407	0.553288	0.554597	0.555819	0.557092	0.557502	0.558032
# 1	0.555949	0.547882	0.548031	0.550044	0.552989	0.555059	0.556286	0.557412
# 1.5	0.554718	0.541145	0.542153	0.545462	0.55059	0.553212	0.555158	0.556406
# 2	0.553621	0.535194	0.536949	0.542157	0.546768	0.551521	0.5537	0.556171
# 2.5	0.552309	0.528409	0.530979	0.536652	0.543674	0.548462	0.552253	0.55507"""),
#             'output':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.55798	0.557633	0.557363	0.557631	0.557984	0.558422	0.558858	0.556351
# 0.5	0.555131	0.551985	0.550766	0.553721	0.555166	0.55637	0.556571	0.546628
# 1	0.551054	0.542597	0.541428	0.547573	0.549427	0.553713	0.554732	0.515321
# 1.5	0.547081	0.534691	0.533338	0.541192	0.545414	0.550986	0.552416	0.482914
# 2	0.542817	0.527145	0.524312	0.534368	0.54026	0.547982	0.550263	0.441759
# 2.5	0.538949	0.518148	0.516175	0.528866	0.53455	0.543737	0.547757	0.413484"""),
#             'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
# 0.1	0.557195	0.555515	0.55595	0.55642	0.557593	0.558196	0.558594	0.558517
# 0.5	0.550896	0.53842	0.541827	0.546705	0.552092	0.554539	0.55549	0.556195
# 1	0.543659	0.517598	0.525267	0.534074	0.54345	0.549529	0.553113	0.553481
# 1.5	0.535836	0.496098	0.506042	0.522621	0.535254	0.54415	0.549795	0.550802
# 2	0.526624	0.477267	0.491125	0.507141	0.525455	0.537201	0.546281	0.547915
# 2.5	0.519397	0.458354	0.474641	0.495698	0.516154	0.530148	0.54085	0.5418"""),
#         },
#     }
# }

# _color_palette = px.colors.qualitative.Plotly
# _layer_color_map = {
#     f"noise_var_{noise}": _color_palette[i]
#     for i,noise in enumerate([0.1,0.5,1.0,1.5,2.0,2.5])
# }

# experiment_versions = ['v43','v44']
# dnn_names = ['dnn2a','dnn2b','dnn2c','dnn2d']

# for k,v in noise_experiments.items():
#     for dnn_number, noise_data in v.items():
#         fig = make_subplots(rows=3, cols=1, subplot_titles=("Noise scaled to input layer size", "Noise scaled to output layer size", "Noise scaled to layer variance"))
#         row = 1
#         for key,data in noise_data.items():

#             df = pd.read_csv(data, sep='\s+').set_index('noise_vars').T
#             print(f"{dnn_number}, {key}")
#             for noise_var in df.columns:
#                 showlegend = True if row == 1 else False
#                 fig.add_trace(go.Scatter(
#                     x=df.index.tolist(),
#                     y=df[noise_var],
#                     name=f"noise = {noise_var}",
#                     marker=dict(color=_layer_color_map[f"noise_var_{noise_var}"]),
#                     showlegend=showlegend,
#                 ),col=1, row=row)
#             row += 1
#         fig.update_layout(
#             width=800,
#             height=1200
#         )
#         base_file_path = f"experiment_plots/{k}/{dnn_number}/"
#         print("saving to" + base_file_path + "noise_accuracy_vs_layers.html")
#         fig.write_html(base_file_path + "noise_accuracy_vs_layers.html")
#         fig.show()