import React from "react";
import { Switch, Route } from "react-router-dom";

class Layout extends React.Component {
    constructor() {
        super();
        this.state = {
            title: "Welcome to React SSR!",
        };
    }

    render() {
        return (
            <div>
                <h1>{ this.state.title }</h1>
            </div>
        );
    }
}

export default Layout;

// import routes from "../routes";

/*
<Switch>
    { routes.map( route => <Route key={ route.path } { ...route } /> ) }
</Switch>
*/