import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Titanic",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """
    Main function to set up the Streamlit app layout.
    """
    cs_sidebar()
    cs_body()
    return None

class SidebarText:
    objetivos = """
    El objetivo de este ejercicio es realizar un análisis exploratorio de datos (EDA) para comprender mejor 
    el conjunto de datos del Titanic.

    Esto implica:

    * **Carga y visualización de los datos:** Importar los datos y familiarizarse con su estructura. 
    * **Resumen estadístico y visualización de datos faltantes:** Identificar datos faltantes y obtener un resumen 
    estadístico básico.
    * **Análisis de variables clave:** Explorar cómo variables específicas, como el sexo, la clase y la edad, afectan 
    la tasa de supervivencia.
    * **Visualización de patrones y relaciones:** Utilizar gráficos para identificar patrones y relaciones en los datos.</small>
    """

    hipotesis = """
    - **Hipótesis 01**: Los pasajeros de primera clase tuvieron la mayor tasa de supervivencia.
    - **Hipótesis 02**: Las mujeres y los niños tuvieron una mayor tasa de supervivencia.
    - **Hipótesis 03**: Los pasajeros que pagaron una tarifa más alta tuvieron una mayor tasa de supervivencia.
    """
# Define the cs_sidebar() function
def cs_sidebar():
    """
    Populate the sidebar with various content sections related to Python.
    """

    st.sidebar.title("Titanic")

    # Título de la aplicación
    st.sidebar.subheader("Objetivos e Hipótesis")

    # Lista de opciones
    opciones = ['Objetivos', 'Hipótesis']

    # Crear el selectbox
    seleccion = st.sidebar.selectbox('Selecciona una opción:', opciones)

    if seleccion == "Objetivos":
        st.sidebar.markdown(SidebarText.objetivos,unsafe_allow_html=True)

    if seleccion == "Hipótesis":
        st.sidebar.markdown(SidebarText.hipotesis, unsafe_allow_html=True)

    return None

def tabla_grafico(df, col1, col2):
    """
    Agrupar dos columnas, cuenta los grupos y los deja en formato df
    """
    df_group = df.groupby([col1, col2]).size().reset_index(name="Count")
    return df_group

def tabla_grafico_1(df, col1, col2):
    """
    Agrupar dos columnas, obtiene los porcentajes y los deja en formato df
    """
    df_group2 = df[[col1,col2]].groupby([col1], as_index=False).mean().sort_values(by="Survived", ascending=False)
    return df_group2

def tabla_grafico_2(df, col1, col2, bins):

  # Agregar una nueva columna al DataFrame con los rangos de la columna específica
  df[col1 + "Range"] = pd.cut(df[col1], bins=bins, right=False)

  # Calcular el conteo de cada grupo y reestructurar los datos
  counts = df.groupby([col2, col1 + "Range"]).size().reset_index(name="Count")

  # Calcular los porcentajes por categoría
  counts["Percentage"] = counts.groupby(col2)["Count"].transform(lambda x: (x / x.sum()))

  # Crear una tabla pivote con los porcentajes
  pivot_counts = counts.pivot_table(values=["Count", "Percentage"], index=col1 + "Range", columns=col2)

  # Eliminar columna extra
  df.drop(col1 + "Range", axis=1, inplace=True)
  return pivot_counts

def tabla_grafico_3(df, col1, col2):

  df['FareBand'] = pd.qcut(df[col1], 4)
  df['FareBand'] = df['FareBand'].astype(str)
  df_colband3= df[['FareBand', col2]].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
  return df_colband3

def tabla_grafico_4(df, col1, col2, col3):

  df["FareBand"] = pd.qcut(df[col1], 4)

  df_e_f = df.groupby([col2, "FareBand", col3]).size().unstack(fill_value=0).reset_index()
  df_e_f["FareBand"] = df_e_f["FareBand"].astype(str)
  df_e_f.columns = [str(x) for x in df_e_f.columns ]
  return df_e_f

def tabla_grafico_5(df, col1, col2, col3):
  # analizaremos los datos del embarque (Embarked) y la clase (Pclass) juntos

  df_e_c = df.groupby([col2, col1, col3]).size().unstack(fill_value=0).reset_index()
  df_e_c = df_e_c.rename(columns = {
      0:'Not_Survived',
      1:'Survived'
  })
  return df_e_c

def mostrar_grafico(df, col1, col2="Survived"):
    # Crear la figura y la gráfica de barras
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x=col1, y='Count', hue=col2, data=df, palette="Set2")

    # Añadir los números encima de las barras
    for p in barplot.patches:
        height = p.get_height()
        if height > 0:
            barplot.annotate(format(height, '.0f'),
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='center',
                             xytext=(0, 9),
                             textcoords='offset points')

    # Título y etiquetas
    plt.title(f"{col2} versus {col1}")
    plt.xlabel(col1)
    plt.ylabel(col2)

    # Ajustar la leyenda
    handles, labels = barplot.get_legend_handles_labels()
    barplot.legend(handles=handles, labels=['No', 'Sí'], title='Sobrevivió')

    # Mostrar la gráfica
    st.pyplot(plt)

def mostrar_grafico_1(df, col1, col2="Survived"):
  # Crear el gráfico de barras
  sns.barplot(x=col1, y=col2, data=df, color="#fdb462")

  # Configurar etiquetas y título
  plt.xlabel(col1)
  plt.ylabel(col2)
  plt.title(f"{col1} versus {col2}")

  # Mostrar el gráfico
  st.pyplot(plt)

def mostrar_grafico_2(df, col1, col2, bins):

  # Configurar el estilo de Seaborn
  sns.set(style="whitegrid")

  # Agregar una nueva columna al DataFrame con los rangos de la columna específica
  df[col1 + "Range"] = pd.cut(df[col1], bins=bins, right=False)

  # Calcular el conteo de cada grupo y reestructurar los datos
  counts = df.groupby([col2, col1+ "Range"]).size().reset_index(name="Count")

  # Calcular los porcentajes por categoría
  counts["Percentage"] = counts.groupby(col2)["Count"].transform(lambda x: (x / x.sum()))

  # Gráfico de barras con Seaborn
  plt.figure(figsize=(8, 4))
  ax = sns.barplot(data=counts, x=col1 + "Range", y="Percentage", hue=col2, palette="Set2")

  # Rotar los ejes x (45 grados) y añadir los valores en cada barra (excluyendo 0%)
  for p in ax.patches:
      if p.get_height() != 0:  # Si el valor no es 0%
          ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()), ha="center",
                      va="center", xytext=(0, 10), textcoords="offset points", fontsize=6)

  # Establecer título y etiquetas de los ejes
  plt.title(f"{col1} versus {col2}")
  plt.xlabel(col1)
  plt.ylabel(col2)
  plt.xticks(rotation=0)  # Rotar etiquetas del eje x para mejor legibilidad
  plt.tight_layout()

  # Mostrar el gráfico
  st.pyplot(plt)

  # Eliminar columna extra del DataFrame
  df.drop(col1 + "Range", axis=1, inplace=True)

def mostrar_grafico_3(df, col2,col3):

  # Crear el gráfico de barras
  sns.barplot(x= col3, y= col2, data=df, color="#fdb462")

  # Configurar etiquetas y título
  plt.xlabel(f"Rango de {col3}")
  plt.ylabel(col2)
  plt.title(f"{col2} versus {col3}")

  # Mostrar el gráfico
  st.pyplot(plt)

def mostrar_grafico_4(df, col1, col2):

  # Crear el histograma usando matplotlib
  fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)

  # Flatten the axs array for easy iteration
  axs = axs.flatten()

  for i, embarked in enumerate(df[col2].unique()):
      df_embarked = df[df[col2] == embarked]
      x = df_embarked["FareBand"]
      y0 = df_embarked["0"]
      y1 = df_embarked["1"]

      bars1 = axs[i].bar(x, y0, label="No Sobrevivieron", alpha=0.7, color="#66c2a4")
      bars2 = axs[i].bar(x, y1, bottom=y0, label="Sobrevivieron", alpha=0.7, color="#fdb462")

      # Añadir etiquetas con los valores encima de las barras
      for bar in bars1:
          yval = bar.get_height()
          axs[i].text(bar.get_x() + bar.get_width()/2., yval + 2, "%d" % int(yval), ha="center", va="bottom", fontsize=10)
      for bar in bars2:
          yval = bar.get_height()
          axs[i].text(bar.get_x() + bar.get_width()/2., yval + bar.get_y() + 2, "%d" % int(yval), ha="center", va="bottom", fontsize=10)

      axs[i].set_title(f"Distribución por Rango de Tarifa para Embarque en {embarked}")
      axs[i].set_xlabel(f"Rango de {col1}")
      axs[i].set_ylabel("Número de Pasajeros")
      axs[i].legend()

  plt.tight_layout()
  st.pyplot(plt)


def mostrar_grafico_5(df, col1, col2, col3):
    # Crear una nueva columna con el total de pasajeros
    df["Total"] = df["Survived"] + df["Not_Survived"]

    # Crear una nueva columna con la tasa de supervivencia
    df["Survival_Rate"] = df[col3] / df["Total"]

    # Graficar los datos
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x=col2, y="Survival_Rate", hue=col1, ci=None, palette="Set2")

    # Añadir etiquetas y título
    plt.title(f"Tasa de Supervivencia por Puerto de {col2} y {col2}")
    plt.xlabel(col2)
    plt.ylabel("Tasa de Supervivencia")
    plt.legend(title=col1)

    # Mostrar el gráfico
    st.pyplot(plt)

# Define the cs_body() function
def cs_body():
    """
    Create content sections for the main body of the
    Streamlit cheat sheet with Python examples.
    """
    # Este es un título principal

    st.title("Descripción del Desafío de Kaggle")

    st.markdown("""
    El Desafío del Titanic en Kaggle es uno de los concursos más populares para principiantes en el campo del 
    análisis de datos y el aprendizaje automático. 

    El objetivo es construir un modelo predictivo que determine
    si un pasajero sobrevivió o no al hundimiento del Titanic en función de las características disponibles
    sobre los pasajeros.

    > **Nota**: Puedes encontrar el desafío en el siguiente enlace: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
    """)

    logo_url = ("https://historia.nationalgeographic.com.es/medio/2023/06/20/the-steamship-titanic-rmg-bhc3667_00000000"
                "_9b5bd117_230620084335_1280x775.jpg")
    st.image(logo_url, width=400)

    st.header("Análisis de Datos")

    # Cargar el DataFrame desde un archivo CSV
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

    st.markdown(""" 
    ### Datos del Titanic

    A continuación se muestra el DataFrame cargado desde un archivo CSV:
    """)

    st.dataframe(df.head())

    st.markdown("## Análisis Exploratorio de Datos")

    # Ver cantidad de sobrevivientes
    sobrevivientes = df.loc[df["Survived"] == 1]

    if st.button("Sobrevivientes"):
        suma_sobrevivientes = sobrevivientes["Survived"].sum()
        st.write(f"Número de sobrevivientes: {suma_sobrevivientes}")

    # Lista de opciones
    opciones = ["Tabla", "Gráfico"]

    # Seleccionar las columnas para agrupar
    columnas_validas = [x for x in df.columns if x not in ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]]
    default_option = "Selecciona la columna"
    col1 = st.selectbox(default_option, [default_option] + columnas_validas)
    col2 = "Survived"


    if col1 != default_option:

        if col1 in ["Sex", "SibSp"]:
            resultado = tabla_grafico(df, col1, col2)
            st.write("Resultado:")
            st.write(resultado)
        elif col1 in ["Age"]:
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            resultado_2 = tabla_grafico_2(df, col1, col2, bins)
            st.write("Resultado:")
            st.write(resultado_2)
        elif col1 in ["Fare"]:
            resultado_3 = tabla_grafico_3(df, col1, col2)
            st.write("Resultado:")
            st.write(resultado_3)

        else:
            resultado_1 = tabla_grafico_1(df, col1, col2)
            st.write("Resultado:")
            st.write(resultado_1)

        if col1 in ["Sex", "SibSp"]:
            resultado = tabla_grafico(df, col1, col2)
            mostrar_grafico(resultado, col1)

        elif col1 in ["Age"]:
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            mostrar_grafico_2(df, "Age", "Survived", bins)

        elif col1 in ["Fare"]:
            resultado_3 = tabla_grafico_3(df, col1, col2)
            mostrar_grafico_3(resultado_3,"Survived","FareBand")

        else:
            resultado_1 = tabla_grafico_1(df, col1, col2)
            mostrar_grafico_1(resultado_1,col1)

        if col1 in ["Sex"]:
            st.markdown("""
            El gráfico anterior ilustra claramente la influencia del sexo en la tasa de supervivencia de los pasajeros.
            En total, habían más hombres que mujeres entre los pasajeros. La tasa de supervivencia es considerablemente
            mayor en mujeres que en hombres. La diferencia en la tasa de supervivencia entre géneros podría deberse a 
            las políticas de evacuación del Titanic, que priorizaron a mujeres y niños. En resumen, el gráfico resalta
            una clara disparidad en la supervivencia entre hombres y mujeres durante el desastre del Titanic, con una
            notable ventaja de supervivencia para las mujeres.
            """)
        elif col1 in ["Age"]:
            st.markdown("""
            Podemos observar en el gráfico el porcentaje de sobrevivientes del Titanic por rangos de edad.
            La mayor tasa de supervivencia se encuentra en el rango de edad de [0-10] años. A medida que aumenta la 
            edad, la probabilidad de supervivencia disminuye paulatinamente. El rango de edad de [30-40] años es el 
            que contaba con más pasajeros y muestra los porcentajes más altos tanto de no sobrevivientes como de 
            sobrevivientes, con una ligera mayoría de no sobrevivientes. En los adultos, la probabilidad de
            supervivencia es más equilibrada, mientras que los ancianos presentan una mayor tasa de mortalidad.
            En conclusión, hubo una mayor probabilidad de supervivencia en los más jóvenes, seguida de una tasa más 
            equilibrada entre los adultos y una notable disminución en los ancianos. Esta distribución puede reflejar 
            factores como las prioridades en el rescate (por ejemplo, "mujeres y niños primero") y la capacidad física 
            para enfrentar la situación de emergencia.
            """)

        elif col1 in ["Fare"]:
            st.markdown("""
            Con los datos proporcionados, podemos concluir que la probabilidad de supervivencia era mayor para aquellos 
            que habían pagado una tarifa más elevada.
            """)

        elif col1 in ["Pclass"]:
            st.markdown("""
            El gráfico muestra claramente que la clase fue un factor determinante en la supervivencia de los pasajeros. 
            Los pasajeros de primera clase tuvieron la tasa de supervivencia más alta, seguidos por los de segunda 
            clase, mientras que los de tercera clase tuvieron la menor tasa de supervivencia.
            """)

        elif col1 in ["SibSp"]:
            st.markdown("""
            A través de la tabla y gráfico, podemos observar que los pasajeros que más sobrevivieron fueron aquellos que 
            viajaban solos; sin embargo, estos no tenían la tasa más alta de supervivencia, debido a que había más datos 
            de ellos. Los de mayor probabilidad de sobrevivir fueron los que tenían 1 o 2 familiares a bordo, ya fueran 
            hermanos o cónyuges, posiblemente porque era más fácil ayudarse mutuamente. En cambio, los que viajaban solos 
            tenían una probabilidad intermedia de supervivencia, y esta probabilidad disminuía cuando tenían más de 2 
            familiares, lo cual podría deberse a la dificultad de mantenerse unidos en la emergencia.
            """)

        elif col1 in ["Parch"]:
            st.markdown("""
            Similar al caso anterior, las familias compuestas por padres e hijos tuvieron una mayor tasa de 
            supervivencia. En particular, aquellos que viajaban entre 1 y 3 familiares presentaron una mayor 
            probabilidad de sobrevivir, probablemente debido a una mejor coordinación. Los pasajeros que viajaban 
            solos tenían una probabilidad intermedia de supervivencia, mientras que esta probabilidad disminuía 
            considerablemente si había más de 4 familiares a bordo. Estos resultados indican que viajar con uno o pocos 
            familiares (padres o hijos) aumentaba las probabilidades de supervivencia, posiblemente debido al apoyo y la 
            colaboración en situaciones de emergencia.
            """)

        elif col1 in ["Embarked"]:
            st.markdown("""
            Para facilitar el análisis, aclaramos que las letras representan los siguientes puertos de embarque:
            * C es Cherburgo
            * Q es Queenstown
            * S es Southampton.
            Observamos en la tabla y en el gráfico la tasa de supervivencia más alta se encuentra en los pasajeros que 
            embarcaron en Cherburgo (C), una probabilidad intermedia en los de Queenstown (Q) y la más baja en los de 
            Southampton (S). Aunque actualmente la información es limitada, realizaremos análisis adicionales para 
            obtener conclusiones más detalladas.
            """)

    st.markdown("Análisis complementario del Puerto de Embarque con la Tarifa y la Clase")

    columnas_escogidas = [x for x in df.columns if x in ["Fare", "Pclass"]]
    default_option = "Selecciona la columna"
    col1 = st.selectbox(default_option, [default_option] + columnas_escogidas)
    col2 = "Embarked"
    col3 = "Survived"

    if col1 != default_option:

        if col1 in ["Fare"]:
            resultado_4 = tabla_grafico_4(df, col1, col2, col3)
            st.write("Resultado:")
            st.write(resultado_4)

        if col1 in ["Pclass"]:
            resultado_5 = tabla_grafico_5(df, col1, col2, col3)
            st.write("Resultado:")
            st.write(resultado_5)

        if col1 in ["Fare"]:
            resultado_4 = tabla_grafico_4(df, col1, col2, col3)
            mostrar_grafico_4(resultado_4, "Fare", "Embarked")

        if col1 in ["Pclass"]:
            resultado_5 = tabla_grafico_5(df, col1, col2, col3)
            mostrar_grafico_5(resultado_5, "Pclass", "Embarked", "Survived")

        if col1 in ["Fare"]:
            st.markdown("""
            El gráfico muestra claramente que los pasajeros de Cherburgo tendieron a pagar tarifas más altas, mientras que 
            los de Queenstown pagaron principalmente las tarifas más bajas. En Southampton, la tendencia fue a pagar la 
            segunda tarifa más baja. Es importante destacar que en Southampton subieron más pasajeros que en los otros dos 
            puertos.
            """)

        if col1 in ["Pclass"]:
            st.markdown("""
            Podemos observar que en Cherburgo hay una mayor cantidad de pasajeros de primera clase, mientras que en 
            Queenstown predominan los pasajeros de segunda clase. Por otro lado, Southampton tiene la mayor cantidad de 
            pasajeros de tercera clase. Esto sugiere que el puerto de embarque influye en la composición de clases a bordo, 
            y respalda la idea de que los pasajeros de primera clase tenían mayores probabilidades de sobrevivir que los de 
            las demás clases.
            """)

    if st.button("Conclusión"):
        st.markdown("""
            
        **Hipótesis 01**: Los pasajeros de primera clase tuvieron la mayor tasa de supervivencia.
        
        > En los análisis realizados previamente, se observó que se otorgó prioridad a las personas de esta clase durante la 
        emergencia.
        
        **Hipótesis 02**: Las mujeres y los niños tuvieron una mayor tasa de supervivencia.
        
        > Esta hipótesis también se confirma, probablemente debido a la política de "mujeres y niños primero".
        
        **Hipótesis 03**: Los pasajeros que pagaron una tarifa más alta tuvieron una mayor tasa de supervivencia.
        
        > Efectivamente, estos pasajeros fueron priorizados durante la emergencia.
        
        
        En cuanto a los demás aspectos analizados, se puede decir que, respecto a las familias, incluyendo cónyuges, hermanos, 
        padres e hijos, aquellas que eran pequeñas tenían más probabilidades de sobrevivir, ya que pudieron manejar mejor la 
        crisis. Por último, el puerto de embarque fue un factor influyente, ya que determinó las clases de los pasajeros y, 
        por lo tanto, la prioridad que recibieron durante la emergencia.
        
        """)

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()




