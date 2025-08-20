import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Konfiguracja strony
st.set_page_config(
    page_title="Eksploracja Danych Ankiety Powitalnej",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funkcja do wczytania danych
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('35__welcome_survey_cleaned.csv', sep=';')
        df = df.replace('', np.nan)
        return df
    except FileNotFoundError:
        st.error("Nie znaleziono pliku '35__welcome_survey_cleaned.csv'. Upewnij się, że plik znajduje się w tym samym folderze co aplikacja.")
        st.stop()

def clean_data(df):
    """Funkcja do czyszczenia danych"""
    # Zastąp puste stringi wartościami NaN
    df = df.replace('', np.nan)
    df = df.replace(' ', np.nan)
    return df

def create_age_distribution(df):
    """Wykres rozkładu wieku"""
    age_data = df['age'].dropna()
    if len(age_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Rozkład przedziałów wiekowych", height=400)
        return fig
    
    age_counts = age_data.value_counts().sort_index()
    fig = px.bar(
        x=age_counts.index, 
        y=age_counts.values,
        title="Rozkład przedziałów wiekowych",
        labels={'x': 'Przedział wiekowy', 'y': 'Liczba osób'},
        color=age_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False, height=400)
    return fig

def create_education_pie(df):
    """Wykres kołowy poziomu wykształcenia"""
    edu_data = df['edu_level'].dropna()
    if len(edu_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Rozkład poziomu wykształcenia", height=400)
        return fig
    
    edu_counts = edu_data.value_counts()
    fig = px.pie(
        values=edu_counts.values,
        names=edu_counts.index,
        title="Rozkład poziomu wykształcenia"
    )
    fig.update_layout(height=400)
    return fig

def create_hobbies_heatmap(df):
    """Wykres popularności hobby"""
    hobby_cols = [col for col in df.columns if col.startswith('hobby_')]
    if not hobby_cols:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych o hobby", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Popularność hobby", height=400)
        return fig
    
    hobby_data = df[hobby_cols].sum().sort_values(ascending=True)
    hobby_data = hobby_data[hobby_data > 0]  # Usuń hobby z zerowymi wartościami
    
    if len(hobby_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych o hobby", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Popularność hobby", height=400)
        return fig
    
    fig = px.bar(
        x=hobby_data.values,
        y=[col.replace('hobby_', '').replace('_', ' ').title() for col in hobby_data.index],
        orientation='h',
        title="Popularność hobby",
        labels={'x': 'Liczba osób', 'y': 'Rodzaj hobby'},
        color=hobby_data.values,
        color_continuous_scale='plasma'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_learning_preferences(df):
    """Preferencje sposobów nauki"""
    learning_cols = [col for col in df.columns if col.startswith('learning_pref_')]
    if not learning_cols:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych o preferencjach nauki", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Preferencje sposobów nauki", height=400)
        return fig
    
    learning_data = df[learning_cols].sum().sort_values(ascending=False)
    learning_data = learning_data[learning_data > 0]  # Usuń metody z zerowymi wartościami
    
    if len(learning_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych o preferencjach nauki", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Preferencje sposobów nauki", height=400)
        return fig
    
    fig = px.bar(
        x=[col.replace('learning_pref_', '').replace('_', ' ').title() for col in learning_data.index],
        y=learning_data.values,
        title="Preferencje sposobów nauki",
        labels={'x': 'Sposób nauki', 'y': 'Liczba osób'},
        color=learning_data.values,
        color_continuous_scale='blues'
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_motivation_radar(df):
    """Wykres radarowy motywacji"""
    motivation_cols = [col for col in df.columns if col.startswith('motivation_')]
    if not motivation_cols:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych o motywacji", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Rozkład motywacji", height=500)
        return fig
    
    motivation_data = df[motivation_cols].sum()
    motivation_data = motivation_data[motivation_data > 0]  # Usuń motywacje z zerowymi wartościami
    
    if len(motivation_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych o motywacji", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Rozkład motywacji", height=500)
        return fig
    
    categories = [col.replace('motivation_', '').replace('_', ' ').title() for col in motivation_data.index]
    values = motivation_data.values.tolist()
    values.append(values[0])  # Zamknij wykres radarowy
    categories.append(categories[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Motywacja',
        line_color='rgb(32, 122, 199)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values)])
        ),
        title="Rozkład motywacji",
        height=500,
        showlegend=False
    )
    return fig

def create_experience_vs_age(df):
    """Związek między wiekiem a doświadczeniem"""
    df_clean = df.dropna(subset=['age', 'years_of_experience'])
    
    if len(df_clean) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Związek między wiekiem a doświadczeniem zawodowym", height=500)
        return fig
    
    # Stwórz wykres słupkowy zamiast sunburst dla lepszej czytelności
    crosstab = pd.crosstab(df_clean['age'], df_clean['years_of_experience'])
    
    fig = px.bar(
        crosstab,
        title="Związek między wiekiem a doświadczeniem zawodowym",
        labels={'value': 'Liczba osób', 'index': 'Przedział wiekowy'},
        height=500
    )
    fig.update_layout(xaxis_tickangle=45)
    return fig

def create_industry_analysis(df):
    """Analiza branż"""
    industry_data = df['industry'].dropna()
    industry_data = industry_data[industry_data.str.strip() != '']  # Usuń puste stringi
    
    if len(industry_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Brak danych o branżach",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16
        )
        fig.update_layout(
            title="Top 10 najczęstszych branż",
            height=500,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    
    industry_counts = industry_data.value_counts().head(10)
    
    # Użyj wykresu słupkowego zamiast treemap dla lepszej czytelności
    fig = px.bar(
        x=industry_counts.values,
        y=industry_counts.index,
        orientation='h',
        title="Top 10 najczęstszych branż",
        labels={'x': 'Liczba osób', 'y': 'Branża'},
        color=industry_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500, showlegend=False)
    return fig

def create_sweet_salty_analysis(df):
    """Analiza preferencji słodkie vs słone"""
    sweet_salty_data = df['sweet_or_salty'].dropna()
    
    if len(sweet_salty_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Preferencje: Słodkie vs Słone", height=400)
        return fig
    
    counts = sweet_salty_data.value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="Preferencje: Słodkie vs Słone",
        color_discrete_sequence=colors
    )
    fig.update_layout(height=400)
    return fig

def create_gender_distribution(df):
    """Rozkład płci"""
    gender_data = df['gender'].dropna()
    
    if len(gender_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Rozkład płci", height=400)
        return fig
    
    gender_labels = {0.0: 'Kobieta', 1.0: 'Mężczyzna'}
    gender_counts = gender_data.map(gender_labels).value_counts()
    
    fig = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        title="Rozkład płci",
        color_discrete_sequence=['#FF69B4', '#87CEEB']
    )
    fig.update_layout(height=400)
    return fig

def main():
    st.title("📊 Eksploracja Danych Ankiety Powitalnej")
    st.markdown("---")
    
    # Wczytanie danych
    df = load_data()
    df = clean_data(df)
    
    # Sidebar z filtrami
    st.sidebar.title("🔧 Filtry")
    
    # Filtr wieku
    age_options = ['Wszystkie'] + sorted([x for x in df['age'].dropna().unique() if pd.notna(x)])
    selected_age = st.sidebar.selectbox("Przedział wiekowy:", age_options)
    
    # Filtr wykształcenia
    edu_options = ['Wszystkie'] + sorted([x for x in df['edu_level'].dropna().unique() if pd.notna(x)])
    selected_edu = st.sidebar.selectbox("Wykształcenie:", edu_options)
    
    # Filtr branży
    industry_options = ['Wszystkie'] + sorted([x for x in df['industry'].dropna().unique() if pd.notna(x) and str(x).strip() != ''])
    selected_industry = st.sidebar.selectbox("Branża:", industry_options)
    
    # Filtr płci
    gender_options = ['Wszystkie', 'Kobieta', 'Mężczyzna']
    selected_gender = st.sidebar.selectbox("Płeć:", gender_options)
    
    # Filtr ulubionych zwierząt
    animals_options = ['Wszystkie'] + sorted([x for x in df['fav_animals'].dropna().unique() if pd.notna(x) and str(x).strip() != ''])
    selected_animals = st.sidebar.selectbox("Ulubione zwierzęta:", animals_options)
    
    # Aplikowanie filtrów
    filtered_df = df.copy()
    
    if selected_age != 'Wszystkie':
        filtered_df = filtered_df[filtered_df['age'] == selected_age]
    
    if selected_edu != 'Wszystkie':
        filtered_df = filtered_df[filtered_df['edu_level'] == selected_edu]
    
    if selected_industry != 'Wszystkie':
        filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
    
    if selected_gender != 'Wszystkie':
        gender_val = 0.0 if selected_gender == 'Kobieta' else 1.0
        filtered_df = filtered_df[filtered_df['gender'] == gender_val]
    
    if selected_animals != 'Wszystkie':
        filtered_df = filtered_df[filtered_df['fav_animals'] == selected_animals]
    
    # Wyświetlenie podstawowych statystyk
    st.subheader("📈 Podstawowe Statystyki")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Liczba respondentów", len(filtered_df))
    
    with col2:
        if len(filtered_df) > 0 and not filtered_df['age'].dropna().empty:
            avg_age_group = filtered_df['age'].dropna().mode().iloc[0]
        else:
            avg_age_group = "Brak danych"
        st.metric("Najczęstszy przedział wiekowy", avg_age_group)
    
    with col3:
        if len(filtered_df) > 0 and not filtered_df['edu_level'].dropna().empty:
            most_common_edu = filtered_df['edu_level'].dropna().mode().iloc[0]
        else:
            most_common_edu = "Brak danych"
        st.metric("Najczęstsze wykształcenie", most_common_edu)
    
    with col4:
        sweet_data = filtered_df['sweet_or_salty'].dropna()
        if len(sweet_data) > 0:
            sweet_percentage = (sweet_data == 'sweet').mean() * 100
        else:
            sweet_percentage = 0
        st.metric("% preferujących słodkie", f"{sweet_percentage:.1f}%")
    
    st.markdown("---")
    
    # Główne wykresy
    if len(filtered_df) > 0:
        # Pierwszy rząd wykresów
        st.subheader("👥 Dane demograficzne")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_age_distribution(filtered_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_gender_distribution(filtered_df), use_container_width=True)
        
        # Drugi rząd wykresów
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_education_pie(filtered_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_sweet_salty_analysis(filtered_df), use_container_width=True)
        
        # Trzeci rząd wykresów
        st.subheader("🎯 Zainteresowania i motywacja")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_hobbies_heatmap(filtered_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_learning_preferences(filtered_df), use_container_width=True)
        
        # Czwarty rząd wykresów
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_motivation_radar(filtered_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_experience_vs_age(filtered_df), use_container_width=True)
        
        # Piąty rząd - analiza branż
        st.subheader("💼 Analiza zawodowa")
        st.plotly_chart(create_industry_analysis(filtered_df), use_container_width=True)
        
        # Analiza korelacji
        st.subheader("🔗 Analiza Korelacji")
        
        hobby_cols = [col for col in filtered_df.columns if col.startswith('hobby_')]
        motivation_cols = [col for col in filtered_df.columns if col.startswith('motivation_')]
        learning_cols = [col for col in filtered_df.columns if col.startswith('learning_pref_')]
        
        if hobby_cols or motivation_cols or learning_cols:
            all_binary_cols = hobby_cols + motivation_cols + learning_cols
            correlation_data = filtered_df[all_binary_cols].corr()
            
            fig = px.imshow(
                correlation_data,
                title="Mapa korelacji między hobby, motywacją i preferencjami nauki",
                color_continuous_scale='RdBu_r',
                aspect='auto',
                labels=dict(color="Korelacja")
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Sekcja z danymi szczegółowymi
        st.subheader("🔍 Szczegółowe Dane")
        
        # Wybór kolumn do wyświetlenia
        all_columns = list(filtered_df.columns)
        default_cols = ['age', 'edu_level', 'fav_animals', 'industry', 'sweet_or_salty', 'years_of_experience']
        available_defaults = [col for col in default_cols if col in all_columns]
        
        selected_columns = st.multiselect(
            "Wybierz kolumny do wyświetlenia:",
            all_columns,
            default=available_defaults
        )
        
        if selected_columns:
            st.dataframe(filtered_df[selected_columns], use_container_width=True)
        
        # Możliwość pobrania danych
        st.subheader("💾 Pobierz Dane")
        csv = filtered_df.to_csv(index=False, sep=';')
        st.download_button(
            label="📥 Pobierz przefiltrowane dane jako CSV",
            data=csv,
            file_name='filtered_survey_data.csv',
            mime='text/csv'
        )
        
        # Statystyki opisowe
        st.subheader("📊 Statystyki Opisowe")
        
        # Statystyki dla zmiennych kategorycznych
        categorical_cols = ['age', 'edu_level', 'fav_animals', 'fav_place', 'industry', 'sweet_or_salty', 'years_of_experience']
        available_categorical = [col for col in categorical_cols if col in filtered_df.columns]
        
        if available_categorical:
            selected_cat_col = st.selectbox("Wybierz zmienną do analizy:", available_categorical)
            if selected_cat_col:
                cat_stats = filtered_df[selected_cat_col].value_counts()
                st.write(f"**Rozkład wartości dla '{selected_cat_col}':**")
                st.dataframe(cat_stats.to_frame().rename(columns={selected_cat_col: 'Liczba'}))
    
    else:
        st.warning("⚠️ Brak danych spełniających wybrane kryteria filtrowania.")
        st.info("Spróbuj zmienić filtry w panelu bocznym.")

if __name__ == "__main__":
    main()