import streamlit as st
import pydeck as pdk

# from streamlit_echarts import st_pyecharts
# from pyecharts import options as opts
# from pyecharts.charts import Line, Scatter, Timeline
import vrptw

import pandas as pd
import numpy as np

# Page settings
st.set_page_config(
    page_title="Visualization of VRP",
    page_icon="❄️️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This app is used to visualize the results of vehicle routing problems!"
    }
)

# Set up main page
col1, col2 = st.columns((6, 1))
col1.title("🐻‍❄ Visualization of VRP 🐻‍❄️")
col2.image("pictures/snowflake_dcr_multi.png", width= 120)
st.sidebar.image("pictures/bear_snowflake_hello.png")
action = st.sidebar.radio("What action would you like to take?", ("Add Vehicles 🐧️",
                                                                  "Add Customers ☃️",
                                                                  "Add Depots 💧",
                                                                  "Visualize Routes 🐻‍❄"))

st.markdown("Generates routes and visualization results for vehicle routing problem.")
st.markdown("This sample is provided for reference purposes only.")


# Gets version info
def get_version_info():
    version_file = "VERSION.md"
    version_list = []
    # Read version file
    try:
        with open(version_file, "r", encoding='utf-8') as fin:
            for line in fin:
                version_list.append(line)
    finally:
        return version_list

with st.expander("Version Information"):
    for record in get_version_info():
        st.write(record)

##########
def get_paths():
    paths = vrptw.run()
    df = pd.DataFrame(paths, columns=["path"])
    return df

def show_routes():
    df = get_paths()

    view_state = pdk.ViewState(latitude=31.213901, longitude=121.396759, zoom=10)

    line_layer = pdk.Layer(
        type="PathLayer",
        data=df,
        pickable=True,
        get_color='[0, 255, 0]',
        width_scale=20,
        width_min_pixels=2,
        get_path="path",
        get_width=5,
    )
    r = pdk.Deck(layers=[line_layer], initial_view_state=view_state, tooltip={"text": "{name}"})
    return r


# 生成随机的车辆行驶路径
# def showPath():
#     # 生成随机的车辆行驶路径
#     x = np.random.randint(0, 100, 10)
#     y = np.random.randint(0, 100, 10)

#     # 创建时间轴
#     timeline = Timeline()
#     # 创建折线图和散点图
#     for i in range(len(x)):
#         scatter = (
#             Scatter()
#             .add_xaxis(x[:i+1].tolist())
#             .add_yaxis('', y[:i+1].tolist())
#             .set_series_opts(opts.LabelOpts(is_show=False))
#         )
#         line = (
#             Line()
#             .add_xaxis(x[:i+1].tolist())
#             .add_yaxis('', y[:i+1].tolist(), is_symbol_show= False)
#             .set_global_opts(
#                 xaxis_opts=opts.AxisOpts(min_=0, max_=100, is_show=False),
#                 yaxis_opts=opts.AxisOpts(min_=0, max_=100, is_show=False),
#             ).set_series_opts(opts.LabelOpts(is_show=False))
#         )    
#         # 添加到时间轴
#         timeline.add(line.overlap(scatter), str(i))
#     return timeline

if action == "Add Vehicles 🐧️":
    st.subheader("❄️ Add Vehicles! ❄️")

elif action == "Add Customers ☃️":
    st.subheader("❄️ Add Customers! ❄️")

elif action == "Add Depots 💧":
    st.subheader("❄️ Add Depots! ❄️")

elif action == "Visualize Routes 🐻‍❄":
    st.subheader("❄️ Visualize Routes! ❄️")
    option = st.selectbox(
     'What is your main objective of vehicle routing problem?',
     ('Minimize the number of vehicles', 'Minimize the total cost'))
    # st.write('Your favorite color is ', option)

    options = st.multiselect(
     'What are the constraints you need?',
     ['Multiple vehicle types', 'Multiple parking lots', 'Elastic overload', 'Distribution priority'],
     ['Multiple vehicle types'])
    # st.write('You selected:', options)

    st.pydeck_chart(show_routes())
    # st_pyecharts(showPath())

    
