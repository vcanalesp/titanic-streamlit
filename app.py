# Import convention
import streamlit as st
import pandas as pd

st.title("Doberman")
st.header("Perritos")
st.subheader("Nut y Seth")
st.text("Guau sin formato")
st.markdown("__*Guau en Cursiva y negrita*__")

logo_url = "https://vevico.wordpress.com/wp-content/uploads/2020/11/doberman.png?w=718"
st.image(logo_url, width=200)
st.caption('Doberman de orejas caidas')
st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.write("Guau que admite diferentes formatos")
st.code('for i in range(8): foo()')

st.button('Hit me')

def greet(name):
    st.write(f'Hola, {name}!')

name = st.text_input('Introduce tu nombre')
if st.button('Saludar', on_click=greet, args=(name,)):
    greet(name)

if st.button('Haz clic aquí'):
    st.write('¡Hola soy dobbie!')

# Crear un DataFrame
data = {
    'Columna1': range(1, 21),
    'Columna2': [x * 2 for x in range(1, 21)],
    'Columna3': [x ** 2 for x in range(1, 21)]
}
my_dataframe = pd.DataFrame(data)

# Mostrar el DataFrame completo
st.dataframe(my_dataframe)

# Mostrar las primeras 10 filas del DataFrame (en este caso solo hay 5)
st.table(my_dataframe.iloc[0:10])

# Mostrar un JSON
st.json({'foo':'bar', 'fu':'ba'})

# Mostrar una métrica
st.metric(label="Temp", value="273 K", delta="1.2 K")

st.image('./Doberman.jpg')

col1, col2 = st.columns(2)
col1.write('Nut')
col2.write('Seth')

# Three columns with different widths
col1, col2, col3 = st.columns([2,2,1])
# col1 is wider

# Using 'with' notation:
with col1:
    st.write('Hola soy Nut')

with col2:
    st.write('Hola soy Seth')

with col3:
    if st.button('Haz clic'):
        st.write('¡Hola soy dobbie!')

# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Pestaña 1", "Pestaña 2"])
tab1.write("Esto es de la pestaña 1")
tab2.write("Esto es de la pestaña 2")

# You can also use "with" notation:
with tab1:
  st.radio('Seleccione uno:', [1, 2])

with tab2:
    st.button('Hello')

# Stop execution immediately:
#st.stop()
# Rerun script immediately:
#st.experimental_rerun()

# Group multiple widgets:
with st.form(key='my_form'):
    username = st.text_input('Username')
    password = st.text_input('Password')
    st.form_submit_button('Login')

# Show different content based on the user's email address.
#if st.user.email == 'jane@email.com':
#  display_jane_content()
#elif st.user.email == 'adam@foocorp.io':
#   display_adam_content()
#else:
#   st.write("Please contact us to get access!")


#st.data_editor('Edit data', data)
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')
#st.download_button('On the dl', data)
#st.camera_input("一二三,茄子!")
st.color_picker('Pick a color')

# Use widgets' returned values in variables
#for i in range(int(st.number_input('Num:'))): foo()
#if st.sidebar.selectbox('I:',['f']) == 'f': b()
#my_slider_val = st.slider('Quinn Mallory', 1, 88)
#st.write(slider_val)

# Disable widgets to remove interactivity:
st.slider('Pick a number', 0, 100, disabled=True)

# Insert a chat message container.
with st.chat_message("user"):
   st.write("Hello 👋")
   #st.line_chart(np.random.randn(30, 3))

# Display a chat input widget.
st.chat_input("Say something")

# Add rows to a dataframe after
# showing it.
#element = st.dataframe(df1)
#element.add_rows(df2)

# Add rows to a chart after
# showing it.
#element = st.line_chart(df1)
#element.add_rows(df2)

st.echo()
with st.echo():
    st.write('Code will be executed and printed')

# Replace any single element.
#element = st.empty()
#element.line_chart(...)
#element.text_input(...)  # Replaces previous.

#sert out of order.
#elements = st.container()
#elements.line_chart(...)
#st.write("Hello")
#elements.text_input(...)  # Appears above "Hello".
#
#st.help(pandas.DataFrame)
#st.get_option(key)
#st.set_option(key, value)
#st.set_page_config(layout='wide')
#st.experimental_show(objects)
#st.experimental_get_query_params()
#st.experimental_set_query_params(**params)

#st.experimental_connection('pets_db', type='sql')
#conn = st.experimental_connection('sql')
#conn = st.experimental_connection('snowpark')

#class MyConnection(ExperimentalBaseConnection[myconn.MyConnection]):
#   def _connect(self, **kwargs) -> MyConnection:
#       return myconn.connect(**self._secrets, **kwargs)
#   def query(self, query):
#      return self._instance.query(query)

# E.g. Dataframe computation, storing downloaded data, etc.
#@st.cache_data
#def foo(bar):
#  # Do something expensive and return data
#  return data
## Executes foo
#d1 = foo(ref1)
## Does not execute foo
## Returns cached item by value, d1 == d2
#d2 = foo(ref1)
## Different arg, so function foo executes
#d3 = foo(ref2)
## Clear all cached entries for this function
#foo.clear()
## Clear values from *all* in-memory or on-disk cached functions
#st.cache_data.clear()

# E.g. TensorFlow session, database connection, etc.
#@st.cache_resource
#def foo(bar):
#  # Create and return a non-data object
#  return session
## Executes foo
#s1 = foo(ref1)
## Does not execute foo
## Returns cached item by reference, s1 == s2
#s2 = foo(ref1)
## Different arg, so function foo executes
#s3 = foo(ref2)
## Clear all cached entries for this function
#foo.clear()
## Clear all global resources from cache
#st.cache_resource.clear()

# Show a spinner during a process
#with st.spinner(text='In progress'):
#  time.sleep(3)
#  st.success('Done')

# Show and update progress bar
#bar = st.progress(50)
#time.sleep(3)
#bar.progress(100)

#st.balloons()
#st.snow()
st.toast('Mr Stay-Puft')
st.error('Error message')
st.warning('Warning message')
st.info('Info message')
st.success('Success message')
#st.exception(e)

